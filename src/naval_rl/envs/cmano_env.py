"""
CMATNOEnv — Gymnasium environment backed by Command: Modern Operations.

State is parsed from CMANO's full scenario XML using pycmo.FeaturesFromSteam,
giving access to unit positions, weapon loadouts, and sensor contacts.

Observation and action spaces are identical to NavalEnv so existing TD3 agents
and the training loop in train.py are drop-in compatible.

Observation vector layout  (per ship, alice first then bob):
  [ x_m, y_m, course_rad, speed_mpm, health_frac ]   × (n_alice + n_bob)

  positions   metres, local XY frame centred on scenario.center_lat/lon
  course_rad  math convention: 0=East, CCW positive (converted from CMO compass heading)
  speed_mpm   metres per minute (converted from knots)
  health_frac 1.0 = alive, 0.0 = destroyed (unit missing from XML → dead)

Action vector layout (per ship, same order):
  [ speed_fraction, course_rad, fire_fraction, target_fraction ]

Install deps:
  pip install ".[cmano]"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pycmo.lib.features import FeaturesFromSteam

from naval_rl.bridge.coord_transform import (
    compass_to_math_rad,
    knots_to_mpm,
    latlon_to_xy,
    math_rad_to_compass,
)
from naval_rl.bridge.ipc import FileIPC
from naval_rl.rewards.potential_fields import compute_rewards


_FEATURES_PER_SHIP = 5
_ACTIONS_PER_SHIP  = 4
_FIRE_THRESHOLD    = 0.5


class CMATNOEnv(gym.Env):
    """
    Adversarial naval environment using CMANO as the physics / simulation backend.

    Parameters
    ----------
    cfg         : full YAML config dict (see configs/cmano_cat_and_mouse.yaml)
    bridge_dir  : shared IPC directory visible to both this process and CMO
    ipc_timeout : seconds to wait for CMO before raising TimeoutError
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Dict[str, Any],
        bridge_dir: str = "/tmp/cmano_bridge",
        ipc_timeout: float = 120.0,
    ) -> None:
        super().__init__()

        self.cfg       = cfg
        self.ipc       = FileIPC(bridge_dir, timeout=ipc_timeout)
        self.cfg_alice = cfg["reward_alice"]
        self.cfg_bob   = cfg["reward_bob"]

        scenario   = cfg["scenario"]
        self.lat0  = scenario["center_lat"]
        self.lon0  = scenario["center_lon"]
        self.grid_half = cfg.get("grid_half", 100_000.0)

        self.alice_ids: List[str] = [u["unit_id"] for u in cfg["fleet_alice"]]
        self.bob_ids:   List[str] = [u["unit_id"] for u in cfg["fleet_bob"]]
        self.n_alice = len(self.alice_ids)
        self.n_bob   = len(self.bob_ids)

        self._max_speed_kts: Dict[str, float] = {
            u["unit_id"]: u["max_speed_kts"]
            for u in cfg["fleet_alice"] + cfg["fleet_bob"]
        }

        n_total = self.n_alice + self.n_bob
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_total * _FEATURES_PER_SHIP,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low =np.full(n_total * _ACTIONS_PER_SHIP, -np.inf, dtype=np.float32),
            high=np.full(n_total * _ACTIONS_PER_SHIP,  np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.step_count    = 0
        self.episode_count = 0
        self._prev_alive:  Dict[str, bool] = {}
        self._last_unit:   Dict[str, Any]  = {}   # unit_id → last known pycmo Unit
        # Contact GUID maps built from each step's XML, used for attack targeting.
        self._alice_contact_map: Dict[str, str] = {}  # opponent_name → contact_guid
        self._bob_contact_map:   Dict[str, str] = {}

        self._send_config_and_wait()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0
        self._prev_alive = {uid: True for uid in self.alice_ids + self.bob_ids}

        self.ipc.send_reset(self.episode_count)
        state_data = self.ipc.wait_state()
        alice_units, bob_units = self._update_caches(state_data)
        return self._obs(alice_units, bob_units), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        self.step_count += 1

        self.ipc.send_action(self._encode_action(action))
        state_data = self.ipc.wait_state()
        alice_units, bob_units = self._update_caches(state_data)

        obs     = self._obs(alice_units, bob_units)
        rewards = self._rewards(alice_units, bob_units)

        meta       = state_data["meta"]
        alice_dead = all(not u["alive"] for u in alice_units)
        bob_dead   = all(not u["alive"] for u in bob_units)
        terminated = meta.get("terminal", alice_dead or bob_dead)
        truncated  = False

        if terminated:
            if bob_dead and not alice_dead:
                rewards[0] += self.cfg_alice.get("victory_bonus", 200.0)
                rewards[1] -= self.cfg_bob.get("defeat_penalty", 200.0)
            elif alice_dead and not bob_dead:
                rewards[1] += self.cfg_bob.get("victory_bonus", 200.0)
                rewards[0] -= self.cfg_alice.get("defeat_penalty", 200.0)

        self._prev_alive = {u["unit_id"]: u["alive"] for u in alice_units + bob_units}

        info = {
            "alice_alive": sum(u["alive"] for u in alice_units),
            "bob_alive":   sum(u["alive"] for u in bob_units),
            "rare_event":  self._count_kills(alice_units, bob_units) > 0,
        }
        return obs, np.array(rewards, dtype=np.float32), terminated, truncated, info

    def close(self) -> None:
        self.ipc.send_shutdown()

    # ------------------------------------------------------------------
    # State parsing (pycmo FeaturesFromSteam)
    # ------------------------------------------------------------------

    def _update_caches(
        self, state_data: Dict[str, Any]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse state, update caches, return ordered unit-state dicts for both fleets.

        Two modes:
        - "full": XML parsed with FeaturesFromSteam; contact GUID maps refreshed.
        - "fast": lightweight JSON from ScenEdit_GetUnit; cached contact maps reused.
        """
        if state_data.get("mode") == "fast":
            return self._update_caches_fast(state_data["fast"])
        return self._update_caches_full(state_data["xml"])

    def _update_caches_full(self, xml: str) -> Tuple[List[Dict], List[Dict]]:
        features = FeaturesFromSteam(xml, "Alice")

        alive_alice = {u.Name: u for u in features.get_side_units("Alice")}
        alive_bob   = {u.Name: u for u in features.get_side_units("Bob")}

        for uid, u in {**alive_alice, **alive_bob}.items():
            self._last_unit[uid] = u

        try:
            sides     = features.get_sides()
            alice_idx = sides.index("Alice")
            bob_idx   = sides.index("Bob")
            self._alice_contact_map = {
                c.Name: c.ID for c in features.get_side_contacts(alice_idx) if c.Name
            }
            self._bob_contact_map = {
                c.Name: c.ID for c in features.get_side_contacts(bob_idx) if c.Name
            }
        except (ValueError, IndexError, KeyError):
            pass

        alice_units = self._build_unit_list(self.alice_ids, alive_alice)
        bob_units   = self._build_unit_list(self.bob_ids,   alive_bob)
        return alice_units, bob_units

    def _update_caches_fast(self, fast: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        # Build alive dicts from the lightweight per-unit JSON blobs.
        def _side_alive(unit_list):
            result = {}
            for u in unit_list:
                if not u.get("dead"):
                    result[u["unit_id"]] = u
            return result

        alive_alice = _side_alive(fast.get("alice_units", []))
        alive_bob   = _side_alive(fast.get("bob_units",   []))

        # Update last-unit cache with plain dicts (position fallback on death).
        for uid, u in {**alive_alice, **alive_bob}.items():
            self._last_unit[uid] = u  # plain dict on fast path, Unit namedtuple on full path

        alice_units = self._build_unit_list_fast(self.alice_ids, alive_alice)
        bob_units   = self._build_unit_list_fast(self.bob_ids,   alive_bob)
        return alice_units, bob_units

    def _build_unit_list(
        self,
        unit_ids: List[str],
        alive_dict: Dict[str, Any],
    ) -> List[Dict]:
        """Build unit-state dicts from pycmo Unit namedtuples (full XML mode)."""
        result = []
        for uid in unit_ids:
            if uid in alive_dict:
                u = alive_dict[uid]
                result.append({
                    "unit_id":     uid,
                    "lat":         float(u.Lat or self.lat0),
                    "lon":         float(u.Lon or self.lon0),
                    "heading_deg": float(u.CH  or 0.0),
                    "speed_kts":   float(u.CS  or 0.0),
                    "alive":       True,
                    "damage_pct":  0.0,
                })
            else:
                last = self._last_unit.get(uid)
                lat = (float(last["lat"]) if isinstance(last, dict) else
                       float(last.Lat) if last else self.lat0)
                lon = (float(last["lon"]) if isinstance(last, dict) else
                       float(last.Lon) if last else self.lon0)
                hdg = (float(last.get("heading_deg", 0.0)) if isinstance(last, dict) else
                       float(last.CH or 0.0) if last else 0.0)
                result.append({
                    "unit_id":     uid,
                    "lat":         lat,
                    "lon":         lon,
                    "heading_deg": hdg,
                    "speed_kts":   0.0,
                    "alive":       False,
                    "damage_pct":  100.0,
                })
        return result

    def _build_unit_list_fast(
        self,
        unit_ids: List[str],
        alive_dict: Dict[str, Any],
    ) -> List[Dict]:
        """Build unit-state dicts from lightweight ScenEdit_GetUnit dicts (fast mode)."""
        result = []
        for uid in unit_ids:
            if uid in alive_dict:
                u = alive_dict[uid]
                result.append({
                    "unit_id":     uid,
                    "lat":         float(u.get("lat", self.lat0)),
                    "lon":         float(u.get("lon", self.lon0)),
                    "heading_deg": float(u.get("heading_deg", 0.0)),
                    "speed_kts":   float(u.get("speed_kts",   0.0)),
                    "alive":       True,
                    "damage_pct":  float(u.get("damage_pct", 0.0)),
                })
            else:
                last = self._last_unit.get(uid)
                lat = (float(last["lat"]) if isinstance(last, dict) else
                       float(last.Lat) if last else self.lat0)
                lon = (float(last["lon"]) if isinstance(last, dict) else
                       float(last.Lon) if last else self.lon0)
                hdg = (float(last.get("heading_deg", 0.0)) if isinstance(last, dict) else
                       float(last.CH or 0.0) if last else 0.0)
                result.append({
                    "unit_id":     uid,
                    "lat":         lat,
                    "lon":         lon,
                    "heading_deg": hdg,
                    "speed_kts":   0.0,
                    "alive":       False,
                    "damage_pct":  100.0,
                })
        return result

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs(self, alice_units: List[Dict], bob_units: List[Dict]) -> np.ndarray:
        parts = []
        for u in alice_units + bob_units:
            x, y       = latlon_to_xy(u["lat"], u["lon"], self.lat0, self.lon0)
            course_rad = compass_to_math_rad(u["heading_deg"])
            speed_mpm  = knots_to_mpm(u["speed_kts"])
            health     = max(0.0, 1.0 - u["damage_pct"] / 100.0)
            parts.extend([x, y, course_rad, speed_mpm, health])
        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _rewards(
        self, alice_units: List[Dict], bob_units: List[Dict]
    ) -> List[float]:
        def _arrays(units):
            pos, vel = [], []
            for u in units:
                x, y = latlon_to_xy(u["lat"], u["lon"], self.lat0, self.lon0)
                pos.append([x, y])
                h = math.radians(u["heading_deg"])
                s = knots_to_mpm(u["speed_kts"])
                vel.append([s * math.sin(h), s * math.cos(h)])
            return np.array(pos, dtype=np.float32), np.array(vel, dtype=np.float32)

        pos_A, vel_A = _arrays(alice_units)
        pos_B, vel_B = _arrays(bob_units)

        r_A, r_B = compute_rewards(
            pos_A, pos_B, vel_A, vel_B,
            self.cfg_alice, self.cfg_bob,
            self.grid_half,
        )

        for i, u in enumerate(bob_units):
            if self._prev_alive.get(u["unit_id"], True) and not u["alive"]:
                r_A += self.cfg_alice.get("kill_reward", 100.0)
                r_B[i] -= self.cfg_bob.get("death_penalty", 100.0)
        for i, u in enumerate(alice_units):
            if self._prev_alive.get(u["unit_id"], True) and not u["alive"]:
                r_B += self.cfg_bob.get("kill_reward", 100.0)
                r_A[i] -= self.cfg_alice.get("death_penalty", 100.0)

        return [float(r_A.mean()), float(r_B.mean())]

    def _count_kills(
        self, alice_units: List[Dict], bob_units: List[Dict]
    ) -> int:
        return sum(
            1 for u in alice_units + bob_units
            if self._prev_alive.get(u["unit_id"], True) and not u["alive"]
        )

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------

    def _encode_action(self, action: np.ndarray) -> Dict:
        action = np.asarray(action, dtype=np.float32).reshape(-1, _ACTIONS_PER_SHIP)
        act_A  = action[:self.n_alice]
        act_B  = action[self.n_alice:]

        alive_bob   = [uid for uid in self.bob_ids   if self._prev_alive.get(uid, True)]
        alive_alice = [uid for uid in self.alice_ids if self._prev_alive.get(uid, True)]

        def _encode(unit_ids, acts, opponent_ids, contact_map):
            out = []
            n_opp = len(opponent_ids)
            for uid, a in zip(unit_ids, acts):
                fire_frac   = float(a[2])
                target_frac = float(a[3])
                target_id   = None
                if n_opp > 0 and fire_frac > _FIRE_THRESHOLD:
                    idx = int(np.clip(
                        round(0.5 * (target_frac + 1) * (n_opp - 1)), 0, n_opp - 1
                    ))
                    name = opponent_ids[idx]
                    # Prefer contact GUID (required by ScenEdit_AttackContact);
                    # fall back to unit name if contact not yet in picture.
                    target_id = contact_map.get(name, name)
                out.append({
                    "unit_id":     uid,
                    "speed_kts":   float(np.clip(a[0], 0.0, 1.0)) * self._max_speed_kts[uid],
                    "heading_deg": math_rad_to_compass(float(a[1])),
                    "fire":        fire_frac > _FIRE_THRESHOLD,
                    "target_id":   target_id,
                })
            return out

        return {
            "alice_actions": _encode(self.alice_ids, act_A, alive_bob,   self._alice_contact_map),
            "bob_actions":   _encode(self.bob_ids,   act_B, alive_alice, self._bob_contact_map),
        }

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _send_config_and_wait(self) -> None:
        scenario = self.cfg["scenario"]
        self.ipc.send_config({
            "center_lat":       self.lat0,
            "center_lon":       self.lon0,
            "step_seconds":     scenario.get("step_seconds", 30),
            "time_compression": scenario.get("time_compression", 300),
            "max_steps":        self.cfg.get("training", {}).get("max_steps_per_episode", 500),
            "alice_units":      self.cfg["fleet_alice"],
            "bob_units":        self.cfg["fleet_bob"],
        })
        self.ipc.wait_ready()
