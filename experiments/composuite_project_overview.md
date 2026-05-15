# CompoSuite × LIBERO × openpi — Project Overview

## What this project is

A pipeline for **scripted demonstration collection on the LIBERO-CompoSuite
task suite**, in a form that can be fed directly into the **openpi VLA
fine-tuning stack** (Pi-zero / Pi-0.5).

The local LIBERO checkout is a **fork that ports CompoSuite's compositional
task generator into LIBERO** (4 robots × 4 objects × 4 obstacles × 4 task
objectives). It already wires up the env, observations, language strings,
and per-objective shaped rewards. What it does **not** ship is demonstrations
for those CompoSuite-style tasks — and openpi's LIBERO fine-tuning recipe
needs `LeRobotDataset`-shaped demos. Closing that gap is the point of this
project.

## Why this matters

LIBERO is the de-facto VLA benchmark (Pi-zero/0.5, OpenVLA, Octo are all
reported on it), and `pi05_libero` is a turnkey checkpoint. The CompoSuite
extension makes LIBERO **compositional** — instead of a fixed handful of
suites, it generates a 4×4×4×4 grid of tasks where train/test splits can
restrict any axis (e.g. "withhold IIWA", "withhold plate"). That gives a
real compositional-generalization eval bed for VLAs. But all of that is
useless without per-task demonstrations to fine-tune on, since the
CompoSuite-generated tasks aren't in any pretrained VLA's training mix.

## Scope of the demo collector

A single Panda robot with the **OSC_POSE** controller, 256×256 agentview +
wrist cameras, fingers truncated to 2-D gripper qpos. We currently target
**three task objectives × three object types**, with a single
`obstacle="none"`:

| Object   | pick_and_place | trash_can | shelf |
|----------|----------------|-----------|-------|
| box      | ✅ working      | ✅ working | ✅ working |
| dumbbell | ✅ working      | ✅ working | ✅ working |
| plate    | ✅ working (new)| ⏳         | ⏳         |

The `box` row was the original pilot (proves the openpi data contract end
to end). `dumbbell` extended it to a non-trivial geometry. `plate` was the
hardest case and required **physical-fixture work** (see below).

## Per-task scripted recipes

All recipes are open-loop OSC_POSE waypoint state machines parameterized by
object pose, goal pose, and a small set of per-object grip constants. Each
phase has a target eef pose, target quat, gripper command, position
tolerance, and step cap. Most phases early-exit on tolerance; `close` /
`open` phases run for a fixed hold to let the gripper actually act.

- **box** — top-down 4-finger pinch: `approach_above → descend → close →
  lift → move_over → descend_target → open → retract`. Drop heights chosen
  per objective: `pick_and_place=table+0.04`, `trash_can=table+0.20`
  (gravity drops it through the rim), `shelf=table+0.21` (just above shelf
  top surface at `table+0.17`).
- **dumbbell** — same skeleton, different grip offsets and a slightly
  taller lift to clear the side weights.
- **plate** — **side grasp** with a horizontal wrist (`target_quat =
  [0.5,0.5,0.5,0.5]`), gripping the disc's vertical rim. Required new
  static fixtures (see next section) and a much longer phase pipeline:
  `orient_wrist → rotate_above → descend_outside → slide_in →
  rise_to_grasp → close → slide_off → lift_carry → traverse_y → align_x →
  descend_shelf2 → open → lift_clear → retract`. The `slide_off` phase is
  the one closed-loop step (re-aims every tick to track the disc as it's
  dragged off the source shelf).

## Environment changes (`LIBERO/libero/libero/envs/composuite/`)

- **`_check_success` patches** for `trash_can` (xy near can opening, below
  rim height) and `shelf` (xy aligned with shelf, z within ±10 cm of the
  top surface at `table+0.17`). Verified `False` at reset and `True` after
  manually placing the object.
- **Plate-disc tuning** in `composuite_plate.xml`: r=0.060, half-height
  0.009, density=60, friction `"20.0 2.0 0.2"`, plus
  `bottom`/`top`/`horizontal_radius` sites for grasp targeting.
- **Two static shelves for the plate task** (`composuite_plate_shelf.xml`):
  source shelf the disc rests on, destination shelf the disc gets placed
  on. Both spawned automatically when `object_type="plate"`. The plate
  goal (`right_bin_pos`) is overridden to the destination shelf top.
- Plate `_check_success` uses a tighter z tolerance (0.025 m) so success
  only fires when the disc is *actually settled* on the destination shelf
  (not while being carried at altitude).

## Demo data format (matches `LiberoInputs` exactly)

One `.npz` per successful episode:

| key           | shape              | dtype   | meaning                                         |
|---------------|--------------------|---------|-------------------------------------------------|
| `image`       | `(T, 256, 256, 3)` | uint8   | agentview RGB                                   |
| `wrist_image` | `(T, 256, 256, 3)` | uint8   | `robot0_eye_in_hand` RGB                        |
| `state`       | `(T, 8)`           | float32 | `[eef_pos(3), axis_angle(3), gripper_qpos[:2]]` |
| `actions`     | `(T, 7)`           | float32 | `[Δpos(3), Δaxis_angle(3), gripper(1)]`         |
| `task`        | scalar string      | `<U…`   | auto-generated language instruction             |

Plus an MP4 of the agentview frames at `videos/{task}_ep{i}_{succ|fail}.mp4`.

The `libero` conda env is Python 3.8 (`lerobot` requires ≥ 3.10), so the
collector writes raw NPZs in the LeRobot field schema. A small ingestion
step inside the openpi `.venv` (Python ≥ 3.11) converts these to a
`LeRobotDataset` via `add_frame(...)` (see
`convert_composuite_npz_to_lerobot.py`).

## Validated results so far

- **box × {pick_and_place, trash_can, shelf}**: 5/5 each, ~1–3 s/attempt.
- **dumbbell × {pick_and_place, trash_can, shelf}**: working in the same
  pipeline.
- **plate × pick_and_place (shelf-to-shelf)**: 5/5 with horizontal side
  grasp, closed-loop slide_off, two-shelf carry, ~5.8 s/attempt.
- Pi-0.5 zero-shot eval harness (`composuite_smoke.py`) runs end to end
  against the same env builder.

## What's left to do

1. **Plate × {trash_can, shelf}** — extend the side-grasp recipe to the
   other two objectives (mostly retargeting the destination, plus a higher
   release for trash_can).
2. **Visual sanity-check pass** — render an MP4 per task variant and
   eyeball for collisions / weird grips before scaling up.
3. **Scale to ≥ 50 demos per task** for VLA fine-tuning. Cost is cheap
   (~5 s per plate attempt, ~2 s per box/dumbbell attempt).
4. **NPZ → LeRobotDataset ingestion** under the openpi `.venv` (script
   already exists; just run it across the full collected set).
5. **Fine-tune Pi-0.5 (`pi05_libero`)** on the resulting dataset and
   evaluate via `composuite_smoke.py`.
6. **Wire the remaining objectives** (`push`) and **restricted-axis
   benchmark presets** (`\Panda`, `∩plate`, etc.) into
   `CompoSuiteBenchmark` so we can run real compositional-generalization
   evals after fine-tuning.
7. **Optional**: switch the gripper from Robotiq85 to Rethink (paper-
   faithful CompoSuite) and add visual/texture randomization.

## Key files

- `LIBERO/libero/libero/envs/composuite/composuite_env.py` — env, two-shelf
  wiring, plate goal override, success criteria.
- `LIBERO/libero/libero/envs/composuite/objects.py` — `CompoSuiteBox`,
  `CompoSuiteHollowBox`, `CompoSuiteDumbbell`, `CompoSuitePlate`,
  `PlateShelf`.
- `LIBERO/libero/libero/assets/composuite_objects/*.xml` — MJCF assets.
- `openpi/experiments/composuite_collect.py` — scripted phase pipelines for
  all object/objective combinations.
- `openpi/experiments/composuite_smoke.py` — Pi-zero/0.5 eval harness.
- `openpi/experiments/convert_composuite_npz_to_lerobot.py` — NPZ →
  `LeRobotDataset` converter.
- `openpi/experiments/composuite_pilot_feasibility.md`,
  `composuite_pilot_results.md`,
  `composuite_plate_summary.md` — design + result notes per milestone.

## One-line reproduction

```bash
source /home/amy/miniconda3/etc/profile.d/conda.sh && conda activate libero
cd /home/amy/Projects/openpi
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
python experiments/composuite_collect.py \
    --tasks Panda,box,none,pick_and_place \
            Panda,dumbbell,none,pick_and_place \
            Panda,plate,none,pick_and_place \
    --target-successes-per-task 5 --max-attempts-per-task 8 \
    --max-steps 2500 --out-dir data/composuite/pilot
```
