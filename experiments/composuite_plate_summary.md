# CompoSuite Plate Task — Project Summary

## Goal

Build a scripted demonstration collector for a **side-grasp plate transfer**
task in LIBERO/CompoSuite that:

1. Picks a thin disc ("plate") off a static shelf using a **horizontal-wrist
   side grasp** (fingers vertical, gripping the disc's vertical rim).
2. Carries the disc and **places it on top of a second shelf**.
3. Produces clean, replayable demonstrations (`.npz` + MP4) suitable for
   downstream **VLA fine-tuning** (openpi LeRobot pipeline).

The plate variant is harder than the existing box / dumbbell CompoSuite
tasks because the disc is too thin to top-grasp reliably on the table —
it requires a side-pinch on the rim, which in turn forced the addition
of a physical shelf fixture.

## What's been done

### Environment changes (`LIBERO/libero/libero/envs/composuite/`)

- New static fixture: `composuite_plate_shelf.xml`
  (wooden box, half-extents `(0.040, 0.045, 0.060)`, top at `table_z + 0.120`).
- `PlateShelf` class added to `objects.py`; takes a `name` arg so two
  instances can coexist.
- `composuite_env.py`:
  - **Two** shelves are spawned for the plate task: a source shelf the
    disc rests on, and a destination shelf placed at `(-0.15, 0.28)`.
  - Plate goal (`right_bin_pos`) overridden to the destination
    shelf-top: `(-0.15, 0.28, table_z + 0.122)`.
  - Plate disc XML (`composuite_plate.xml`) tuned for stable side
    grasp: r=0.060, half-height=0.009, density=60, friction
    `"20.0 2.0 0.2"`, with bottom/top/horizontal_radius sites.
  - Source shelf positioned so the disc cantilevers in `-x` toward the
    robot (`shelf_x_offset = +0.020`, disc spawn z = shelf_top + 0.002).
  - `_check_success` for plate uses tighter z tolerance (0.025 m)
    so success only fires when the disc is settled on the destination
    shelf, not while being carried at altitude.

### Scripted demo collector (`openpi/experiments/composuite_collect.py`)

Per-phase plate pipeline (all with horizontal `carry_quat = [0.5,0.5,0.5,0.5]`):

| Phase             | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `orient_wrist`    | Rotate to horizontal side-grasp orientation                |
| `rotate_above`    | Position above the disc                                    |
| `descend_outside` | Drop to grasp height, fingers outside the rim              |
| `slide_in`        | Move sideways to bracket the rim                           |
| `rise_to_grasp`   | Settle finger pads on the disc rim (deep grip, 25 mm inset)|
| `close`           | Hold-close (45 steps) for a firm pinch                     |
| `slide_off`       | **Closed-loop** pull off the source shelf in `-x`          |
| `lift_carry`      | Raise to clear the destination shelf top                   |
| `traverse_y`      | Move `+y` toward the destination at safe x                 |
| `align_x`         | Drift `+x` over the destination shelf                      |
| `descend_shelf2`  | Lower onto destination shelf top                           |
| `open`            | Hold + open the gripper so the disc settles                |
| `lift_clear`      | Straight up so open fingers clear the disc                 |
| `retract`         | Move clear of the work area                                |

Other relevant constants:
- `HOLD_STEPS = {"close": 45, "open": 6}`
- `PHASE_MAX_STEPS = 60`, `P_GAIN = 8.0`, `ROT_GAIN = 1.5`
- `plate_grip_inset = 0.025`, `plate_grip_offset = 0.035`
- Carry-clear height = `shelf2_top + 0.060` (~5 cm clearance over shelf)
- Drop height ≈ `shelf2_top + 0.012` (eef just above disc resting position)
- Language prompt: `"pull the plate off the stand and place it on top of the second stand"`

### Validated behaviour

- Side-grasp pickup: **5/5** success.
- Closed-loop slide off source shelf: clean, no jamming, no re-entry.
- Carry over second shelf with adequate vertical clearance.
- Disc lands on destination shelf at `obj_z ≈ 1.046` (goal `1.022`,
  within tolerance), `obj_to_goal_xy ≈ 0.034`.
- End-to-end plate transfer: **5/5** success (~5.8 s per attempt).

### Lessons learned (preserved for context)

- Panda OSC_POSE z-reach with strict-horizontal wrist tops out
  ~10 cm above the table at distant `(x, y)` — placing on the table
  far from the base is infeasible without a wrist tilt, which itself
  pushes the gripper site **up** at this configuration. Switching the
  destination from "on the table" to "on a second shelf at carry
  height" sidesteps the problem entirely.
- `target_quat = None` means *hold current orientation*, not free.
- Closed-loop slide_off (track disc xy, lead by 2 cm in `-approach_dir`,
  clamp to `slide_off_eef_xy`) was needed to keep the gripper behind
  the disc as it drifts during the pull.

## What's left to do

1. **Visual sanity check**
   Render a video of one full plate transfer episode and eyeball it
   (no shelf collision, disc stays in fingers, lands on shelf2 top).
2. **Full demo batch**
   Generate ≥ 50 successful plate demos (`.npz` + agentview MP4) at
   the existing 256×256 / 8-D state / 7-D action format used by
   `convert_composuite_npz_to_lerobot.py`.
3. **LeRobot conversion**
   Run the existing converter to ingest the `.npz` set into a
   `LeRobotDataset` shard.
4. **VLA fine-tuning**
   Fine-tune the pi-zero / openpi VLA on the plate dataset (mixed
   with the existing box / dumbbell CompoSuite demos as appropriate)
   and evaluate via `composuite_smoke.py`.
5. **Optional polish**
   - Randomise destination shelf `xy` over a small range to broaden
     the demo distribution.
   - Tighten `_check_success` further (require disc velocity ≈ 0 and
     gripper fully open) once the policy is being trained.
   - Add the plate variant to `composuite_smoke.py`'s default task
     specs so eval runs cover it without flag changes.

## Key file paths

- `LIBERO/libero/libero/assets/composuite_objects/composuite_plate.xml`
- `LIBERO/libero/libero/assets/composuite_objects/composuite_plate_shelf.xml`
- `LIBERO/libero/libero/envs/composuite/objects.py` (`PlateShelf`)
- `LIBERO/libero/libero/envs/composuite/composuite_env.py`
  (two-shelf wiring, plate goal override, `_check_success` tightening)
- `openpi/experiments/composuite_collect.py` (scripted phase pipeline)
- `openpi/experiments/composuite_smoke.py` (eval harness)
- `openpi/experiments/convert_composuite_npz_to_lerobot.py` (dataset converter)

## Reproduction

```bash
source /home/amy/miniconda3/etc/profile.d/conda.sh && conda activate libero
cd /home/amy/Projects/openpi
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
python experiments/composuite_collect.py \
    --tasks "Panda,plate,none,pick_and_place" \
    --target-successes-per-task 5 \
    --max-attempts-per-task 8 \
    --max-steps 2500 \
    --out-dir /tmp/collect_lr
```
