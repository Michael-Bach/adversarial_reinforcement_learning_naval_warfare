"""
CMANOClient — high-level Python interface to a running CMANO scenario.

State is parsed from CMANO's full scenario XML using pycmo.FeaturesFromSteam,
giving direct access to PyCMO's Unit, Contact, and weapon named tuples.

Quick start
-----------
    import yaml
    from naval_rl.bridge.client import CMANOClient, UnitCommand

    cfg    = yaml.safe_load(open("configs/cmano_cat_and_mouse.yaml"))
    client = CMANOClient.connect(cfg)

    state = client.reset()
    print(state)                    # SimState — positions, contacts, weapons

    for _ in range(10):
        state = client.step(
            alice=[UnitCommand("Alice-1", speed_kts=20, heading_deg=45)],
            bob  =[UnitCommand("Bob-1",   speed_kts=15, heading_deg=225)],
        )
        print(state.summary())

    # Fire — target_id is resolved to a contact GUID automatically
    state = client.step(
        alice=[UnitCommand("Alice-1", speed_kts=20, heading_deg=45,
                           fire=True, target_name="Bob-1")],
    )

Install deps:  pip install ".[cmano]"
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pycmo.lib.features import Contact, FeaturesFromSteam, Unit

from naval_rl.bridge.coord_transform import latlon_to_xy
from naval_rl.bridge.ipc import FileIPC


# ---------------------------------------------------------------------------
# Command / state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnitCommand:
    """
    Command for a single unit in one time step.

    Parameters
    ----------
    unit_id     : must match a unit_id in the scenario config
    speed_kts   : desired speed in knots
    heading_deg : desired compass heading (0 = North, CW positive)
    fire        : whether to fire at target_name this step
    target_name : display name of the target unit (contact GUID resolved automatically)
    """
    unit_id:     str
    speed_kts:   float          = 0.0
    heading_deg: float          = 0.0
    fire:        bool           = False
    target_name: Optional[str]  = None

    def _to_dict(self, contact_map: Dict[str, str]) -> Dict[str, Any]:
        target_id = None
        if self.fire and self.target_name:
            target_id = contact_map.get(self.target_name, self.target_name)
        return {
            "unit_id":     self.unit_id,
            "speed_kts":   self.speed_kts,
            "heading_deg": self.heading_deg,
            "fire":        self.fire,
            "target_id":   target_id or "",
        }


@dataclass
class UnitState:
    """State of a single unit as reported by CMANO (via PyCMO)."""
    unit_id:     str
    lat:         float
    lon:         float
    heading_deg: float
    speed_kts:   float
    alive:       bool
    x_m:         float = field(default=0.0, repr=False)
    y_m:         float = field(default=0.0, repr=False)
    # Raw PyCMO Unit object — gives weapon loadout, fuel, mount details
    pycmo_unit:  Optional[Unit] = field(default=None, repr=False)

    @property
    def health_frac(self) -> float:
        return 1.0 if self.alive else 0.0

    @property
    def weapon_qty(self) -> int:
        """Total remaining weapon count across all mounts and loadouts."""
        if not self.pycmo_unit:
            return 0
        total = 0
        for m in (self.pycmo_unit.Mounts or []):
            total += sum(w.QuantRemaining or 0 for w in m.Weapons)
        if self.pycmo_unit.Loadout:
            total += sum(w.QuantRemaining or 0 for w in self.pycmo_unit.Loadout.Weapons)
        return total


class SimState:
    """
    Full scenario state at one time step, parsed from CMANO's XML export.

    Attributes
    ----------
    step, game_time_elapsed, terminal, terminal_reason : metadata
    alice_units, bob_units : ordered lists of UnitState
    alice_contacts         : contacts visible to Alice (pycmo Contact named tuples)
    features               : raw FeaturesFromSteam object for advanced access
    """

    def __init__(
        self,
        state_data: Dict[str, Any],
        lat0:       float = 0.0,
        lon0:       float = 0.0,
        alice_ids:  Optional[List[str]] = None,
        bob_ids:    Optional[List[str]] = None,
        last_units: Optional[Dict] = None,
        alice_contact_map: Optional[Dict[str, str]] = None,
        bob_contact_map:   Optional[Dict[str, str]] = None,
    ) -> None:
        meta = state_data["meta"]
        self.step               = meta["step"]
        self.game_time_elapsed  = meta["game_time_elapsed"]
        self.terminal           = meta["terminal"]
        self.terminal_reason    = meta["terminal_reason"]

        self._alice_contact_map: Dict[str, str] = dict(alice_contact_map or {})
        self._bob_contact_map:   Dict[str, str] = dict(bob_contact_map   or {})
        self.features: Optional[Any] = None
        self.alice_contacts: List[Contact] = []

        if state_data.get("mode") == "fast":
            self._init_fast(
                state_data["fast"], lat0, lon0, alice_ids, bob_ids, last_units
            )
        else:
            self._init_full(
                state_data["xml"], lat0, lon0, alice_ids, bob_ids, last_units
            )

    def _init_full(self, xml, lat0, lon0, alice_ids, bob_ids, last_units):
        self.features = FeaturesFromSteam(xml, "Alice")

        alive_alice = {u.Name: u for u in self.features.get_side_units("Alice")}
        alive_bob   = {u.Name: u for u in self.features.get_side_units("Bob")}

        try:
            sides = self.features.get_sides()
            self._alice_contact_map = {
                c.Name: c.ID
                for c in self.features.get_side_contacts(sides.index("Alice"))
                if c.Name
            }
            self._bob_contact_map = {
                c.Name: c.ID
                for c in self.features.get_side_contacts(sides.index("Bob"))
                if c.Name
            }
        except (ValueError, IndexError, KeyError):
            pass

        self.alice_contacts = list(self.features.contacts)
        self.alice_units = self._build_full(alice_ids or list(alive_alice.keys()),
                                            alive_alice, lat0, lon0, last_units)
        self.bob_units   = self._build_full(bob_ids   or list(alive_bob.keys()),
                                            alive_bob,   lat0, lon0, last_units)

    def _init_fast(self, fast, lat0, lon0, alice_ids, bob_ids, last_units):
        def _alive(lst):
            return {u["unit_id"]: u for u in lst if not u.get("dead")}

        alive_alice = _alive(fast.get("alice_units", []))
        alive_bob   = _alive(fast.get("bob_units",   []))

        self.alice_units = self._build_fast(alice_ids or list(alive_alice.keys()),
                                            alive_alice, lat0, lon0, last_units)
        self.bob_units   = self._build_fast(bob_ids   or list(alive_bob.keys()),
                                            alive_bob,   lat0, lon0, last_units)

    @staticmethod
    def _last_pos(last, lat0, lon0):
        if last is None:
            return lat0, lon0, 0.0
        if isinstance(last, dict):
            return float(last["lat"]), float(last["lon"]), float(last.get("heading_deg", 0.0))
        return float(last.Lat), float(last.Lon), float(last.CH or 0.0)

    def _build_full(self, unit_ids, alive_dict, lat0, lon0, last_units):
        result = []
        for uid in unit_ids:
            if uid in alive_dict:
                u = alive_dict[uid]
                ulat = float(u.Lat or lat0)
                ulon = float(u.Lon or lon0)
                x, y = latlon_to_xy(ulat, ulon, lat0, lon0)
                result.append(UnitState(
                    unit_id=uid, lat=ulat, lon=ulon,
                    heading_deg=float(u.CH or 0.0), speed_kts=float(u.CS or 0.0),
                    alive=True, x_m=x, y_m=y, pycmo_unit=u,
                ))
            else:
                last = (last_units or {}).get(uid)
                plat, plon, phdg = self._last_pos(last, lat0, lon0)
                x, y = latlon_to_xy(plat, plon, lat0, lon0)
                result.append(UnitState(
                    unit_id=uid, lat=plat, lon=plon,
                    heading_deg=phdg, speed_kts=0.0,
                    alive=False, x_m=x, y_m=y,
                ))
        return result

    def _build_fast(self, unit_ids, alive_dict, lat0, lon0, last_units):
        result = []
        for uid in unit_ids:
            if uid in alive_dict:
                u = alive_dict[uid]
                ulat = float(u.get("lat", lat0))
                ulon = float(u.get("lon", lon0))
                x, y = latlon_to_xy(ulat, ulon, lat0, lon0)
                result.append(UnitState(
                    unit_id=uid, lat=ulat, lon=ulon,
                    heading_deg=float(u.get("heading_deg", 0.0)),
                    speed_kts=float(u.get("speed_kts", 0.0)),
                    alive=True, x_m=x, y_m=y,
                ))
            else:
                last = (last_units or {}).get(uid)
                plat, plon, phdg = self._last_pos(last, lat0, lon0)
                x, y = latlon_to_xy(plat, plon, lat0, lon0)
                result.append(UnitState(
                    unit_id=uid, lat=plat, lon=plon,
                    heading_deg=phdg, speed_kts=0.0,
                    alive=False, x_m=x, y_m=y,
                ))
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def contact_guid(self, unit_name: str, attacker_side: str = "alice") -> Optional[str]:
        """Look up the contact GUID for a unit name as seen by a given side."""
        m = self._alice_contact_map if attacker_side.lower() == "alice" else self._bob_contact_map
        return m.get(unit_name)

    def summary(self) -> str:
        lines = [
            f"Step {self.step}  t={self.game_time_elapsed:.0f}s"
            + (f"  [{self.terminal_reason}]" if self.terminal else "")
        ]
        for u in self.alice_units:
            lines.append(
                f"  Alice {u.unit_id}: {'ALIVE' if u.alive else 'DEAD '}  "
                f"pos=({u.x_m/1000:.1f},{u.y_m/1000:.1f}) km  "
                f"hdg={u.heading_deg:.0f}°  {u.speed_kts:.1f} kts  "
                f"wpn={u.weapon_qty}"
            )
        for u in self.bob_units:
            lines.append(
                f"  Bob   {u.unit_id}: {'ALIVE' if u.alive else 'DEAD '}  "
                f"pos=({u.x_m/1000:.1f},{u.y_m/1000:.1f}) km  "
                f"hdg={u.heading_deg:.0f}°  {u.speed_kts:.1f} kts  "
                f"wpn={u.weapon_qty}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CMANOClient:
    """
    Step-by-step Python controller for a running CMANO scenario.

    Each call to step() exchanges one message pair with the CMO Lua bridge:
    Python receives the current unit states and contacts, sends commands,
    and CMO advances one time step.

    Parameters
    ----------
    bridge_dir  : shared IPC directory (must match BRIDGE_DIR in Lua)
    ipc_timeout : seconds to wait for CMO before raising TimeoutError
    lat0, lon0  : scenario centre for local XY coordinates in SimState
    """

    def __init__(
        self,
        bridge_dir:  str   = "/tmp/cmano_bridge",
        ipc_timeout: float = 120.0,
        lat0:        float = 0.0,
        lon0:        float = 0.0,
    ) -> None:
        self.ipc      = FileIPC(bridge_dir, timeout=ipc_timeout)
        self.lat0     = lat0
        self.lon0     = lon0
        self._episode = 0
        self._last_unit:   Dict[str, Unit] = {}
        # Contact maps are populated from the last received state
        self._alice_contact_map: Dict[str, str] = {}
        self._bob_contact_map:   Dict[str, str] = {}
        self._alice_ids: Optional[List[str]] = None
        self._bob_ids:   Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        cfg:         Dict[str, Any],
        bridge_dir:  Optional[str]  = None,
        ipc_timeout: float          = 120.0,
    ) -> "CMANOClient":
        """Send scenario config to CMO and wait for the bridge to signal ready."""
        bd  = bridge_dir or cfg.get("bridge_dir", "/tmp/cmano_bridge")
        sc  = cfg.get("scenario", {})
        lat0 = sc.get("center_lat", 0.0)
        lon0 = sc.get("center_lon", 0.0)

        client = cls(bridge_dir=bd, ipc_timeout=ipc_timeout, lat0=lat0, lon0=lon0)
        client._alice_ids = [u["unit_id"] for u in cfg["fleet_alice"]]
        client._bob_ids   = [u["unit_id"] for u in cfg["fleet_bob"]]

        client.ipc.send_config({
            "center_lat":       lat0,
            "center_lon":       lon0,
            "step_seconds":     sc.get("step_seconds", 30),
            "time_compression": sc.get("time_compression", 300),
            "max_steps":        cfg.get("training", {}).get("max_steps_per_episode", 500),
            "alice_units":      cfg["fleet_alice"],
            "bob_units":        cfg["fleet_bob"],
        })
        print(f"Waiting for CMO bridge at {bd} …")
        client.ipc.wait_ready()
        print("CMO bridge ready.")
        return client

    @classmethod
    def launch_and_connect(
        cls,
        cfg:         Dict[str, Any],
        cmo_exe:     str,
        scenario:    str,
        bridge_dir:  Optional[str] = None,
        ipc_timeout: float         = 180.0,
        display:     Optional[str] = None,
    ) -> "CMANOClient":
        """
        Write config, launch CMO as a subprocess, then wait for ready.

        Parameters
        ----------
        display : X11 DISPLAY string for headless Linux/Wine operation, e.g. ":99".
                  If None, uses the current DISPLAY environment variable.
                  Typical headless setup::

                      Xvfb :99 -screen 0 1920x1080x24 &
                      client = CMANOClient.launch_and_connect(
                          cfg, cmo_exe="wine cmo.exe", scenario="bootstrap.scen",
                          display=":99",
                      )

                  On Windows, omit display= entirely; CMO renders to a virtual
                  display adapter if no physical monitor is attached.
        """
        import os
        bd = bridge_dir or cfg.get("bridge_dir", "/tmp/cmano_bridge")
        Path(bd).mkdir(parents=True, exist_ok=True)
        client = cls.connect(cfg, bridge_dir=bd, ipc_timeout=ipc_timeout)

        env = os.environ.copy()
        if display is not None:
            env["DISPLAY"] = display
        env["CMANO_BRIDGE_DIR"] = bd

        print(f"Launching CMO: {cmo_exe} {scenario} /autorun")
        subprocess.Popen([cmo_exe, scenario, "/autorun"], env=env)
        client.ipc.wait_ready()
        return client

    # ------------------------------------------------------------------
    # Episode / step control
    # ------------------------------------------------------------------

    def reset(self) -> SimState:
        """Start a new episode and return the initial state."""
        self._episode += 1
        self.ipc.send_reset(self._episode)
        return self._make_state(self.ipc.wait_state())

    def step(
        self,
        alice: Optional[List[UnitCommand]] = None,
        bob:   Optional[List[UnitCommand]] = None,
    ) -> SimState:
        """
        Send commands and advance one CMO time step.

        Units not listed keep their last issued order (CMO carries it over).
        Pass UnitCommand(unit_id, speed_kts=0) to explicitly stop a unit.
        """
        action = {
            "alice_actions": [c._to_dict(self._alice_contact_map) for c in (alice or [])],
            "bob_actions":   [c._to_dict(self._bob_contact_map)   for c in (bob   or [])],
        }
        self.ipc.send_action(action)
        return self._make_state(self.ipc.wait_state())

    def hold(self) -> SimState:
        """Advance one step with no new commands."""
        return self.step()

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(
        self,
        n_steps:       int,
        alice_policy=None,
        bob_policy=None,
        verbose:       bool = True,
    ) -> List[SimState]:
        """
        Run up to n_steps, optionally applying callable policies.

        alice_policy / bob_policy : callable(SimState) → List[UnitCommand], or None
        Returns all visited states (index 0 = post-reset initial state).
        """
        state  = self.reset()
        states = [state]
        if verbose:
            print(state.summary())
        for _ in range(n_steps):
            if state.terminal:
                break
            state = self.step(
                alice=alice_policy(state) if alice_policy else None,
                bob  =bob_policy(state)   if bob_policy   else None,
            )
            states.append(state)
            if verbose:
                print(state.summary())
        return states

    def close(self) -> None:
        self.ipc.send_shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_state(self, state_data: Dict[str, Any]) -> SimState:
        state = SimState(
            state_data         = state_data,
            lat0               = self.lat0,
            lon0               = self.lon0,
            alice_ids          = self._alice_ids,
            bob_ids            = self._bob_ids,
            last_units         = self._last_unit,
            alice_contact_map  = self._alice_contact_map,
            bob_contact_map    = self._bob_contact_map,
        )
        # Update last-unit cache: prefer pycmo Unit (full mode), fall back to dict (fast mode)
        for u in state.alice_units + state.bob_units:
            if u.pycmo_unit:
                self._last_unit[u.unit_id] = u.pycmo_unit
            elif u.alive:
                self._last_unit[u.unit_id] = {"lat": u.lat, "lon": u.lon, "heading_deg": u.heading_deg}
        # Refresh contact maps only when a full XML step updated them
        if state_data.get("mode") != "fast":
            self._alice_contact_map = state._alice_contact_map
            self._bob_contact_map   = state._bob_contact_map
        return state
