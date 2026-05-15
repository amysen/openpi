# CompoSuite demo-collection pilot — results

## 1. Source patch (`composuite_env._check_success`)

Added two branches to `LIBERO/libero/libero/envs/composuite/composuite_env.py` (`_check_success`, ~L478) — no other core edits:

```python
elif self.objective_type == "trash_can":
    # Object dropped into trash can: xy near can opening, below rim height.
    # Mirrors TrashCanReward (approach_threshold=0.08, rim ~table+0.15).
    goal_pos = self._get_goal_position()
    in_can_xy = np.linalg.norm(object_pos[:2] - goal_pos[:2]) < 0.08
    below_rim = object_pos[2] < self.table_offset[2] + 0.15
    return in_can_xy and below_rim

elif self.objective_type == "shelf":
    # Object placed on shelf: xy aligned with shelf, z above shelf top.
    # Shelf top surface is at table_z + 0.17 (geom_pos 0.16 + half-height 0.01).
    # ShelfReward used 0.16 which targets the underside; check that the object
    # is resting on or near the top surface instead.
    goal_pos = self._get_goal_position()
    shelf_top = self.table_offset[2] + 0.17
    aligned = np.linalg.norm(object_pos[:2] - goal_pos[:2]) < 0.06
    on_shelf = (object_pos[2] > shelf_top - 0.01) and (object_pos[2] < shelf_top + 0.10)
    return aligned and on_shelf
```

Verification (after patch): `_check_success()` returns `False` at reset for all 3 objectives, and returns `True` when the object qpos is moved into the goal pose for each:
```
pick_and_place  reset_success=False  after_place_success=True
trash_can       reset_success=False  after_place_success=True
shelf           reset_success=False  after_place_success=True
```

The shelf threshold (`shelf_height = 0.16`) used by `ShelfReward` in `rewards.py` targets the underside of the shelf top plate; I verified the actual shelf top surface is at `table_z + 0.17` by inspecting the MuJoCo geom (`shelf_g0/g1`: `geom_pos[2]=0.16`, `half_height=0.01`). Documented in the patch comment; the reward function itself was not touched per "smallest possible change".

## 2. Collector

File: `openpi/experiments/composuite_collect.py`

Design (high-level):
- Reuses `_make_env`, `_quat2axisangle`, language generator, and gripper-truncation logic from `composuite_smoke.py`.
- 8-phase OSC_POSE state machine: `approach_above → descend → close → lift → move_over → descend_target → open → retract`, plus a 20-step settling tail so `_check_success` for trash_can/shelf can register after release.
- Proportional waypoint controller: `delta = clip((target − eef) · 8.0, ±1)` mapped into the OSC delta scale, gripper command 1=close / -1=open.
- Per-task drop-z: `pick_and_place=table+0.04`, `trash_can=table+0.20` (release in air, gravity drops it), `shelf=table+0.21`.
- Per-attempt success filter: only saves NPZ if `env._check_success()` (or any in-episode `info["success"]`) is true.
- Output schema (one .npz per successful episode):

| key | shape | dtype |
|---|---|---|
| `image` | `(T, 256, 256, 3)` | uint8 |
| `wrist_image` | `(T, 256, 256, 3)` | uint8 |
| `state` | `(T, 8)` | float32 = `[eef_pos, axisangle, gripper_qpos[:2]]` |
| `actions` | `(T, 7)` | float32 = OSC_POSE deltas + gripper |
| `task` | scalar string | `<U…` |

**Deviation from the feasibility report**: the report planned to write a `LeRobotDataset` directly. The `libero` conda env is Python 3.8.13 and `lerobot` requires ≥ 3.10 (3.0.x≥3.10, 0.5.x≥3.12), so `pip install lerobot` fails. The collector instead writes raw NPZs in the exact LeRobot field schema; a tiny ingestion step inside the openpi `.venv` (Python ≥ 3.11) can later populate a `LeRobotDataset` via `add_frame(...)`. This is a one-time conversion and was deferred per the "do not fine-tune yet" gate.

## 3. Commands run

Patch verification:
```bash
conda activate libero && cd /home/amy/Projects/openpi && export MUJOCO_GL=egl
python - <<'PY'  # reset_success vs after-place success across 3 objectives
... (snippet above)
PY
```

Pilot collection — initial 3-task run:
```bash
conda activate libero && cd /home/amy/Projects/openpi && export MUJOCO_GL=egl
rm -rf data/composuite/pilot
python experiments/composuite_collect.py \
  --tasks Panda,box,none,pick_and_place \
          Panda,box,none,trash_can \
          Panda,box,none,shelf \
  --target-successes-per-task 5 --max-attempts-per-task 30 \
  --max-steps 300 --out-dir data/composuite/pilot
```

Shelf re-run after fixing `shelf` `_check_success` threshold + drop-z:
```bash
rm -rf data/composuite/pilot/episodes/Panda_box_none_shelf* data/composuite/pilot/videos/Panda_box_none_shelf*
python experiments/composuite_collect.py \
  --tasks Panda,box,none,shelf \
  --target-successes-per-task 5 --max-attempts-per-task 30 \
  --max-steps 350 --out-dir data/composuite/pilot
```

## 4. Attempts and successes per task

| Task | Successes | Attempts | Per-episode steps |
|---|---|---|---|
| Panda_box_none_pick_and_place | 5/5 | 5 | 129, 136, 172, 146, 150 |
| Panda_box_none_trash_can | 5/5 | 6 (1 failure: ep01) | 136, 183, 154, 158, 229 |
| Panda_box_none_shelf (after fix) | 5/5 | 5 | 214, 120, 153, 121, 178 |

Wall time per attempt: 1.3–3.2 s. Total pilot wall time including the failed shelf debugging round: ~2 min.

## 5. Dataset location

```
data/composuite/pilot/
├── episodes/    (15 .npz files, ~1.4 GB total uncompressed image data)
├── videos/      (16 .mp4 files: 15 success + 1 trash_can failure)
└── summary.json (last-task-only due to overwrite during shelf re-run)
```

Absolute path: `/home/amy/Projects/openpi/data/composuite/pilot/`

## 6. Sanity check

```
Episodes:      15
Total frames:  2379

Feature schema (first episode):
  image        shape=(129, 256, 256, 3) dtype=uint8
  wrist_image  shape=(129, 256, 256, 3) dtype=uint8
  state        shape=(129, 8)           dtype=float32
  actions      shape=(129, 7)           dtype=float32
  task         shape=()                 dtype=<U45

Sample frame (pick_and_place ep00, t=0):
  state  : [-0.231, 0.010, 0.997, 3.138, 0.058, 0.115, 0.0388, -0.0389]
  actions: [1.000, -1.000, 0.385, 0.0, 0.0, 0.0, -1.000]   # full +x, full -y, partial +z, gripper open
  task   : "pick up the box and place it in the right bin"

Episodes per task:
  Panda_box_none_pick_and_place: n=5, frames=[129, 136, 172, 146, 150]
  Panda_box_none_trash_can:      n=5, frames=[136, 183, 154, 158, 229]
  Panda_box_none_shelf:          n=5, frames=[214, 120, 153, 121, 178]
```

State and action shapes match `LiberoInputs` exactly. `state[0]` gripper components = ±0.0388 = Panda fingers fully open at reset, as expected.

## 7. Saved video paths

`data/composuite/pilot/videos/Panda_box_none_<objective>_ep<NN>_<success|failure>.mp4`, 20 fps, agentview only:
- pick_and_place: ep00–ep04 (all `_success`)
- trash_can: ep00, ep02–ep05 (`_success`); ep01 (`_failure`, kept for inspection)
- shelf: ep00–ep04 (all `_success`, post-fix run)

(The first shelf collection produced 30 `_failure` MP4s that were deleted before the re-run.)

## 8. Where the controller failed and why

**Initial shelf run: 0/30**. Diagnosis: I assumed the shelf top surface was at `table_z + 0.16` (matching the `shelf_height=0.16` constant in `ShelfReward`). The actual MuJoCo geom is at `body_pos.z=0.9 + geom_pos.z=0.16 + half_height=0.01 = 1.07`, with a box of half-height ~0.035 resting on top → `obj_z ≈ 1.105`. My `_check_success` window `[1.03, 1.09]` excluded the actual rest position, and my drop-z `1.065` was below the shelf surface so the gripper jammed against the shelf and never released cleanly. Fixed by:
1. Updating `_check_success(shelf)` to `obj_z ∈ [shelf_top − 0.01, shelf_top + 0.10]` with `shelf_top = table_z + 0.17`.
2. Raising drop-z to `table_z + 0.21`.

**One trash_can failure (ep01)**: the box landed just outside the can rim (`obj_to_goal=0.147`, threshold 0.08). With the current open-loop scripted approach and slight stochasticity in object spawn position, ~1-in-6 trash_can attempts miss. Acceptable; the success filter discards them.

**No other failures observed** for box+none across the three objectives.

## 9. Recommendation

**Proceed to scale this slice to 25–50 demos per task before any controller revision.** Justification:
- Single-attempt success rates: pick_and_place 100%, trash_can ~83%, shelf 100%. With `--max-attempts-per-task=60` the controller would yield ≥50 successes per task in <3 minutes wall-clock.
- All three task videos look clean by inspection (gripper closes on box, lifts, transports, releases at goal). No teleop or human input needed.
- The action distribution is naturally varied across the 5 demos because object spawn xy is randomized in `[-0.25, -0.05] × [-0.3, -0.1]`; the controller adapts deterministically. This is the right kind of variance for behavioral cloning.

**Before scaling, two cheap safety checks:**
1. Spot-check 1 video per task (open the `_success.mp4`s under `data/composuite/pilot/videos/`) to confirm the trajectories look like demonstrations a policy could imitate (smooth, no jitter, gripper open/close at sensible times).
2. Add an `--task-name` or `--out-suffix` argument to the collector so re-runs (e.g. shelf-only) don't overwrite `summary.json`. (Defer; not blocking.)

**Do not proceed to fine-tuning yet.** Scaling to 25–50 demos × 3 tasks (~5 minutes) and one converter pass into a `LeRobotDataset` (run inside `openpi/.venv`) are the next gated steps. Beyond that, fine-tuning still requires: (a) deciding whether to also include `box+none+push` demos so the model retains the one zero-shot capability it has, and (b) selecting LoRA vs full FT, batch size, and step count — out of scope for this pilot.
