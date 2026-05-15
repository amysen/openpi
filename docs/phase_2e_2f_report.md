# Phase 2E + 2F Report — CompoSuite Box / No-Obstacle Demo Dataset

## 1. Exact commands run

```bash
# Collection (LIBERO conda env, MUJOCO_GL=egl, CUDA_VISIBLE_DEVICES=0)
conda activate libero
cd /home/amy/Projects/openpi
for OBJ in pick_and_place trash_can shelf; do
  python experiments/composuite_collect.py \
    --robot Panda --object box --obstacle none --objective $OBJ \
    --num-success 50 --max-attempts 80 --max-steps 350 \
    --out data/composuite/box_objective_v1
done

# (Push attempted with closed-loop controller — see §2/§6 — abandoned; not in dataset.)

# NPZ sanity (LIBERO env)
python experiments/composuite_npz_report.py data/composuite/box_objective_v1/episodes

# Conversion to LeRobot (openpi/.venv, Py 3.11, lerobot 0.1.0)
source /home/amy/Projects/openpi/.venv/bin/activate
python -u experiments/convert_composuite_npz_to_lerobot.py \
  --episodes-dir data/composuite/box_objective_v1/episodes \
  --repo-id composuite_box_objective_v1 --overwrite \
  --image-writer-processes 0 --image-writer-threads 16

# Dataset validation
python /tmp/validate_lerobot.py
```

## 2. Collector changes

Edits to `experiments/composuite_collect.py`:

- New `_carry_height(env, objective)`: shelf → `table_z + 0.32`, others → `table_z + 0.20`. Used for both `lift` and `move_over` (and retract). Fixes the dragged-through-shelf bug — gripper carries the box ~6 cm below itself, so prior `+0.20` lift put the box at `+0.14`, below the shelf top at `+0.17`, and the box was scraping the shelf wall.
- `_drop_target` updated: shelf → `table_z + 0.27` (releases the box on top of the shelf), trash_can → `+0.20`, pick_and_place → `+0.04` (unchanged).
- Refactored `_pickplace_phases(obs, env, objective)` out of `collect_episode`.
- **Push (NEW + DISABLED)**: added `_push_phases` (approach behind, descend to push_z) plus a closed-loop branch in `collect_episode` that re-aims toward `obj_xy + unit·0.05` per step with delta capped at ±0.15. Result: **0/10 successes** because CompoSuite's `PushReward` (`LIBERO/libero/libero/envs/composuite/rewards.py` L160) sets `max_lift_height=0.03` and terminates the episode with `illegal_lift` whenever the box rises >3 cm — the scripted controller cannot stay below this with the OSC_POSE controller's vertical impedance. **Push is unavailable from the scripted controller; we collected only the 3 lift-based tasks.**

## 3. Collection results

| Task | Successes | Attempts | Failures | Mean T | Min T | Max T |
|---|---|---|---|---|---|---|
| pick_and_place | 50 | 50 | 0 | 149.7 | 128 | 193 |
| shelf          | 50 | 50 | 0 | 149.2 | 124 | 204 |
| trash_can      | 50 | 67 | 17 | 172.9 | 128 | 248 |
| **Total**      | **150** | **167** | **17** | **157.3** | **124** | **248** |

- Frames: **23 588** | Episodes NPZ size: **2 772.5 MB** | Videos: 167 MP4s, 12.4 MB.
- Output: `/home/amy/Projects/openpi/data/composuite/box_objective_v1/{episodes,videos,summary.json}`.

## 4. Converter

File: `experiments/convert_composuite_npz_to_lerobot.py`.

Design:

- Builds a `LeRobotDataset.create(repo_id, fps=20, robot_type="panda", features=...)` under `HF_LEROBOT_HOME`.
- Features: `image` (image, 256×256×3), `wrist_image` (image), `state` (f32(8)), `actions` (f32(7)).
- Per NPZ: materialize `images / wrist_images / states / actions` once (critical perf fix — `np.savez_compressed` lazy-decompresses the entire array on every `__getitem__`, so per-frame `d["image"][t]` was re-decompressing ~25 MB ×T per episode). Then `add_frame` in a tight loop, then `save_episode(task=...)`.
- Streams progress every 200 frames with current fps + ETA, and prints `add` / `save` timings per episode.
- Threaded image writer (`--image-writer-threads 16`, processes 0) to avoid 196 KB/image pickle cost; PNG encoding then runs in worker threads in parallel with the main producer.
- `--overwrite` purges any prior dataset dir.

Final throughput: **150 episodes / 23 588 frames in 154 s (≈153 fps)**.

## 5. LeRobot dataset sanity-check

Loaded via `LeRobotDataset("composuite_box_objective_v1")` from `/home/amy/.cache/huggingface/lerobot/composuite_box_objective_v1`:

- `num_episodes=150`, `num_frames=23588`, `fps=20`, `robot_type="panda"`.
- Episode length: min 124, max 248, mean 157.3, sum 23 588.
- Task distribution: 50 / 50 / 50 across the three prompts.
- Features schema:
  - `image`, `wrist_image`: `image` dtype, shape (256, 256, 3), names (height, width, channel).
  - `state`: float32(8), `actions`: float32(7).
  - LeRobot bookkeeping: `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`.
- First sample (`ds[0]`): images returned as `torch.float32` (3, 256, 256); `state` (8,) f32; `actions` (7,) f32; `task = "pick up the box and place it in the right bin"`.
- Sample `state[0]` ≈ `[-0.231, 0.010, 0.997, 3.138, 0.058, 0.115, 0.0389, -0.0389]` (eef_pos(3) + axisangle(3) + gripper_qpos[:2]).
- Sample `actions[0]` = `[1, -1, 0.385, 0, 0, 0, -1]` (OSC_POSE 6-D delta + gripper).
- Schema matches `LeRobotLiberoDataConfig` exactly: `image`, `wrist_image`, `state` (8), `actions` (7), `task`. ✅

## 6. Blockers

- **Push unavailable** from the scripted OSC_POSE controller. CompoSuite's `PushReward.max_lift_height = 0.03` ends the episode the moment the box rises >3 cm; the closed-loop pushing block we added still triggers it. Three options to unblock push:
  1. Override `max_lift_height` in the env config (treat it as a tuning knob for demo collection only).
  2. Collect push demos via teleop / human-in-the-loop instead of the scripted controller.
  3. Use a different low-level controller (e.g. JOINT_POSITION with explicit z-clamp) for push only.
- No other blockers: shelf geometry fixed; conversion fast; LeRobot schema validated.

## 7. Recommendation on fine-tuning

**Proceed to a small (1k–2k step) pi05_libero LoRA / fine-tune sanity run** on `composuite_box_objective_v1`:

- Schema is verified compatible with `LeRobotLiberoDataConfig` — no policy/data plumbing changes needed.
- 150 episodes / 23 588 frames is enough for a "does loss go down and does the policy reproduce a coherent rollout" check, but **not enough to claim CompoSuite generalisation**; treat the run as a wiring/regression test, not a result.
- Recommended scope for the gated run: 3 tasks above only, batch 16, 1k–2k steps, eval rollouts on 5 seeds × 3 tasks.
- **Defer scaling** (more episodes, push, obstacle variants) until after this sanity run lands and we either (a) override `max_lift_height` or (b) decide to teleop the missing skills. Do not include push in any training set until it is collectible reliably.
