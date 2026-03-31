# Streamlit Interactive UI — Design Spec

**Project:** Adversarial Reinforcement Learning in Naval Warfare
**Date:** 2026-03-31
**Author:** Michael Bach

---

## Overview

A Streamlit web app that makes the combat RL system fully interactive: users configure scenarios via a 2D clickable map, launch training, watch agents learn in real time, and replay trained policies from any checkpoint.

---

## Architecture

### Process Model

Training runs as a subprocess (`subprocess.Popen`) spawned by Streamlit. This keeps the compute loop entirely isolated from the UI process — no threading hacks, no GIL contention with PyTorch.

Two files mediate communication:

| File | Writer | Reader | Purpose |
|---|---|---|---|
| `outputs/run_config.json` | Streamlit (Configure page) | `scripts/train.py` | Scenario + hyperparameter config |
| `outputs/live_state.json` | `scripts/train.py` | Streamlit (Training page) | Per-episode state for live animation |

Subprocess lifecycle:
1. User clicks "Start Training" → Streamlit writes `run_config.json`, spawns `train.py --config outputs/run_config.json`, stores PID in `st.session_state`
2. Training loop writes `live_state.json` at end of each episode
3. Streamlit polls every 0.5s via `st.fragment(run_every=0.5)`, reads state, updates UI
4. "Stop Training" sends `SIGTERM` to the subprocess PID

### File Structure

```
app.py                           # Streamlit entry, page routing via st.navigation
pages/
  configure.py                   # Tabs: Battlefield | Ships & Weapons | Hyperparams
  training.py                    # Live animation + reward curves
  replay.py                      # Checkpoint selector + playback
src/naval_rl/
  ui_bridge.py                   # write_live_state(), read_live_state(), schema constants
scripts/
  train.py                       # Extended: accepts run_config.json, writes live_state.json
  replay_rollout.py              # Deterministic rollout → outputs/replay_trajectory.npy
```

### `live_state.json` Schema

```json
{
  "episode": 42,
  "step": 317,
  "ships": [
    {"fleet": "alice", "x": 45.2, "y": 120.1, "heading": 1.3, "alive": true},
    {"fleet": "bob",   "x": 155.0, "y": 80.5, "heading": 4.1, "alive": true}
  ],
  "missiles": [
    {"x": 50.1, "y": 118.3, "heading": 1.1}
  ],
  "rewards": {"alice": 12.4, "bob": -3.1},
  "history": [
    {"episode": 1, "r_alice": 5.2, "r_bob": 2.1}
  ]
}
```

Written atomically (write to `.tmp`, then `os.replace`) to prevent partial reads.

---

## Page 1: Configure

Three tabs.

### Tab 1 — Battlefield Setup

- Plotly scatter chart, 200×200 km grid, fixed aspect ratio
- Radio button above chart: "Place Alice ships" / "Place Bob ships"
- **Click** on chart → places a ship marker at that position (heading defaults to 0°/North)
- Placed ships shown as colored markers with a short arrow indicating heading
- Clicking an existing marker selects it; a heading slider appears to adjust its initial course
- "Clear fleet" button removes all ships for the selected fleet
- Ship count shown as `Alice: 2 ships | Bob: 1 ship`

### Tab 2 — Ship & Weapon Config

- One `st.expander` per placed ship, labelled "Alice Ship 1", "Bob Ship 1", etc.
- Inside each expander:
  - **Hull:** max speed (knots), turn rate (°/step), hull points
  - **Weapons:** one sub-section per weapon slot
    - Weapon type (dropdown: Missile / Torpedo / Gun)
    - Range (km), max salvo size, reload time (steps)
    - AD intercept probability (slider 0–1)

### Tab 3 — Hyperparameters

- Two columns: Alice (left) | Bob (right)
- Per agent: learning rate, batch size, replay buffer size, warmup steps
- Noise: type dropdown (Gaussian / OU / Composite), sigma, decay rate
- Reward weights: sliders for each potential field (Modified Gravity, LJ Supremacy, LJ Formation, Predictive Intercept, Boundary Confinement)
- "Save as YAML" button exports the full config to `configs/custom_<timestamp>.yaml`

**Bottom of page:** "Start Training" button — disabled until ≥1 ship placed per fleet. Clicking writes `run_config.json` and spawns the training subprocess, then navigates to the Training page.

---

## Page 2: Training

### Layout

Two columns: battlefield animation (60%) | metrics panel (40%).

### Battlefield Animation (left)

- Plotly scatter chart, same 200×200 km grid
- Refreshes every 0.5s via `@st.fragment(run_every=0.5)`
- **Alice ships:** blue filled circle + heading arrow
- **Bob ships:** red filled circle + heading arrow
- **Dead ships:** hollow marker, no arrow
- **Missiles in flight:** small × marker in firing fleet's color
- **Ghost trail:** last 20 positions per ship as faded dots (opacity 0.1 → 0.6)
- Episode number and current step displayed above chart

### Metrics Panel (right)

- Two `st.line_chart` plots: Alice episode reward history | Bob episode reward history
- Summary stats below: episodes completed, total kills per fleet, avg episode length
- All data read from `live_state.json["history"]`

### Controls

- "Stop Training" button: sends `SIGTERM`, shows "Training stopped", disables itself
- "Go to Replay" link button: appears once `outputs/` contains at least one `.pt` checkpoint pair

---

## Page 3: Replay

### Checkpoint Selector

- Dropdown listing checkpoint pairs found in `outputs/`: e.g. `ep250`, `ep500`, `final`
- Pairs matched by filename pattern `alice_<tag>.pt` / `bob_<tag>.pt`
- "Load & Run" button: spawns `replay_rollout.py` with selected checkpoint tags, waits for it to write `outputs/replay_trajectory.npy`, then loads for playback

### Playback Controls

- Play / Pause button
- Step Forward / Step Back buttons (advance or rewind one step)
- Speed multiplier slider: 0.1× to 4×
- Step scrubber slider (0 → max steps), synced with playback position

### Battlefield View

- Same Plotly chart with ghost trail enabled by default
- Missile events shown as brief expanded markers (visible for 3 frames)
- Kill events annotated on the scrubber as red tick marks

### Stats Panel (alongside battlefield)

- Per-step reward for Alice and Bob
- Running kill count
- Current step / total steps

### How Playback Works

`replay_rollout.py` loads the selected `.pt` checkpoints, runs one deterministic episode (noise=0, `actor.eval()`), and dumps the full trajectory as a structured numpy array to `outputs/replay_trajectory.npy`. The UI then drives playback entirely from this array — no subprocess needed during playback, just index manipulation on the loaded array.

---

## Modified Training Script (`scripts/train.py`)

Two additions:
1. Accept `--config outputs/run_config.json` (JSON parsed identically to YAML)
2. Call `write_live_state(state)` from `ui_bridge.py` at the end of each episode

`write_live_state` writes atomically and is a no-op if `ui_bridge` is not imported (backwards compatible with YAML-only usage).

---

## Dependencies

New packages to add to `pyproject.toml`:

```
streamlit>=1.33          # st.fragment(run_every=...) requires 1.33+
plotly>=5.18
streamlit-plotly-events  # click events on Plotly charts for ship placement
```

---

## Out of Scope

- Adjusting hyperparameters mid-training (locked once training starts)
- Multi-run comparison dashboards
- Remote/cloud training
