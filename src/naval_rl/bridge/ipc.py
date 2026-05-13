"""
File-based IPC between the Python training process and the CMANO Lua bridge.

State channel  : state.xml  (full scenario XML from ScenEdit_ExportScenarioToXML)
                 state_meta.json  (step index, game time, terminal flag)
Action channel : action.json  (per-unit speed/heading/fire commands)

Flag files act as atomic signals; data files are written before their flags.

File layout in bridge_dir:
  config.json        Python → Lua   scenario config, written once at startup
  config.flag        Python → Lua   signals config.json is ready
  ready.flag         Lua → Python   signals scenario has been built
  reset.json         Python → Lua   episode number for the reset
  reset.flag         Python → Lua   triggers episode reset in Lua
  state.xml          Lua → Python   full scenario XML (parsed by FeaturesFromSteam)
  state_meta.json    Lua → Python   step, game_time_elapsed, terminal, terminal_reason
  state.flag         Lua → Python   signals both state files are ready
  action.json        Python → Lua   per-unit actions
  action.flag        Python → Lua   signals action.json is ready
  shutdown.flag      Python → Lua   signals clean shutdown

For cross-machine setups (CMO on Windows, Python on Linux), point bridge_dir
at a network share visible to both hosts (e.g. a Samba mount).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple


class FileIPC:
    """Thin wrapper around the shared IPC directory."""

    def __init__(self, bridge_dir: str, timeout: float = 60.0) -> None:
        self.dir = Path(bridge_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Startup handshake
    # ------------------------------------------------------------------

    def send_config(self, cfg: Dict[str, Any]) -> None:
        self._atomic_write("config.json", json.dumps(cfg))
        self._flag("config.flag").write_text("1")

    def wait_ready(self) -> None:
        self._wait("ready.flag")
        self._flag("ready.flag").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Per-episode reset
    # ------------------------------------------------------------------

    def send_reset(self, episode: int) -> None:
        # Clear stale state files before signalling reset
        self._flag("state.flag").unlink(missing_ok=True)
        self._atomic_write("reset.json", json.dumps({"episode": episode}))
        self._flag("reset.flag").write_text("1")

    # ------------------------------------------------------------------
    # Per-step exchange
    # ------------------------------------------------------------------

    def wait_state(self) -> Dict[str, Any]:
        """
        Block until Lua writes a new state, then return a state dict.

        Two modes — callers handle both transparently:

        Full XML mode  (mode="full", fires happened or post-reset):
            {"mode": "full", "xml": str, "meta": dict}
            xml is passed to FeaturesFromSteam; contact GUIDs are refreshed.

        Fast mode  (mode="fast", most steps):
            {"mode": "fast", "fast": dict, "meta": dict}
            "fast" contains alice_units/bob_units lists from ScenEdit_GetUnit.
            No XML; Python uses cached contact maps from last full-mode step.
        """
        self._wait("state.flag")
        meta = json.loads(self._path("state_meta.json").read_text(encoding="utf-8"))
        self._flag("state.flag").unlink(missing_ok=True)

        if meta.get("mode") == "fast":
            fast = json.loads(self._path("state_fast.json").read_text(encoding="utf-8"))
            return {"mode": "fast", "fast": fast, "meta": meta}
        else:
            xml = self._path("state.xml").read_text(encoding="utf-8")
            return {"mode": "full", "xml": xml, "meta": meta}

    def send_action(self, action: Dict[str, Any]) -> None:
        self._atomic_write("action.json", json.dumps(action))
        self._flag("action.flag").write_text("1")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def send_shutdown(self) -> None:
        self._flag("shutdown.flag").write_text("1")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path(self, name: str) -> Path:
        return self.dir / name

    def _flag(self, name: str) -> Path:
        return self.dir / name

    def _atomic_write(self, name: str, content: str) -> None:
        tmp = self._path(name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(self._path(name))

    def _wait(self, flag_name: str) -> None:
        deadline = time.monotonic() + self.timeout
        flag = self._flag(flag_name)
        while not flag.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"FileIPC: timed out after {self.timeout}s waiting for "
                    f"'{flag_name}' in {self.dir}"
                )
            time.sleep(0.01)
