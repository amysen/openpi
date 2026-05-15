# CompoSuite Demo / Fine-Tune Pilot — Feasibility Report

## 1. Demo-generation options that exist locally

| Option | Where it lives | Cost to use | Verdict for our 3 tasks |
|---|---|---|---|
| **A. LIBERO teleop** (`LIBERO/scripts/collect_demonstration.py`, `LIBERO/scripts/libero_100_collect_demonstrations.py`) | LIBERO repo | Needs SpaceMouse/keyboard, ~1–2 min/episode by hand | Works but slow; 30–150 episodes = many human-hours |
| **B. Robosuite teleop** (`robosuite/scripts/collect_human_demonstrations.py` + `wrappers/data_collection_wrapper.py`) | robosuite site-packages | Same as A | Same as A |
| **C. Upstream `composuite` PyPI scripted skills** | NOT installed in `libero` env (not in `LIBERO/requirements.txt`); the LIBERO fork's port at `libero/libero/envs/composuite/` reimplements the env without copying the skills | Requires adding a dep + wiring its skills to our env; nontrivial because their skills target the upstream env class | Skip |
| **D. MolmoSpaces planner (CuRobo)** | molmospaces repo | Needs CUDA + setup we don't currently use, and you said don't switch | Skip |
| **E. Custom OSC_POSE waypoint controller** | Doesn't exist; ~150 LOC to write | One file in `experiments/` | **Recommended** |

**Available scene info confirms E is straightforward** — the CompoSuite env exposes `obs["object_pos"]`, `obs["goal_pos"]`, `obs["object_to_goal_pos"]`, `obs["robot0_eef_pos"]`, `obs["robot0_eef_quat"]`, `obs["robot0_gripper_qpos"]`, plus `agentview_image` and `robot0_eye_in_hand_image`. The upstream-style `right_bin_pos`, trash, and shelf body positions are accessible via `env.sim.data.body_xpos[...]` from the env, and `goal_pos` is published to obs already.

## 2. Recommendation

**Option E — custom OSC_POSE waypoint scripted controller**, written as a single file under `openpi/experiments/`, reusing the same env builder that `composuite_smoke.py` already uses (OSC_POSE + dual cameras + 8-D state with the gripper truncation). Bypass HDF5 entirely and write directly to a LeRobot dataset using the same `LeRobotDataset.add_frame(...)` loop that `convert_libero_data_to_lerobot.py` uses.

Rationale:
- Zero dependency on an external scripted-policy package or human teleop.
- The 3 chosen tasks are all `box + none` — the simplest scene; a 6-state machine (above-obj → descend → close → lift → above-target → release) is enough.
- Generates demos in seconds each, so 5 → 50 scaling is cheap and we can throw away failures and resample.
- We already proved the env contract end-to-end at inference time, so the same wrapper produces consumable demos with no further core changes.

## 3. Files found (absolute paths)

```
LIBERO fork:
  /home/amy/Projects/LIBERO/scripts/collect_demonstration.py
  /home/amy/Projects/LIBERO/scripts/libero_100_collect_demonstrations.py
  /home/amy/Projects/LIBERO/libero/libero/envs/composuite/composuite_env.py
  /home/amy/Projects/LIBERO/libero/libero/envs/composuite/rewards.py
  /home/amy/Projects/LIBERO/libero/libero/envs/composuite/objects.py
  /home/amy/Projects/LIBERO/libero/libero/benchmark/composuite_benchmark.py

OpenPI:
  /home/amy/Projects/openpi/examples/libero/convert_libero_data_to_lerobot.py
  /home/amy/Projects/openpi/examples/libero/README.md
  /home/amy/Projects/openpi/examples/libero/main.py
  /home/amy/Projects/openpi/src/openpi/training/config.py        (pi05_libero @ ~line 838)
  /home/amy/Projects/openpi/src/openpi/policies/libero_policy.py (LiberoInputs)
  /home/amy/Projects/openpi/experiments/composuite_smoke.py      (proven env wrapper)

Robosuite (env site-packages):
  /home/amy/miniconda3/envs/libero/lib/python3.8/site-packages/robosuite/wrappers/data_collection_wrapper.py
  /home/amy/miniconda3/envs/libero/lib/python3.8/site-packages/robosuite/scripts/collect_human_demonstrations.py
  /home/amy/miniconda3/envs/libero/lib/python3.8/site-packages/robosuite/devices/{keyboard,spacemouse}.py
```

## 4. Exact data format OpenPI expects for LIBERO fine-tuning

`convert_libero_data_to_lerobot.py` is the canonical example. It reads the openvla `*_no_noops` RLDS shards but the actual sink is a generic `LeRobotDataset`:

```python
dataset = LeRobotDataset.create(
    repo_id=REPO_NAME, robot_type="panda", fps=10,
    features={
        "image":       {"dtype":"image","shape":(256,256,3),"names":["height","width","channel"]},
        "wrist_image": {"dtype":"image","shape":(256,256,3),"names":["height","width","channel"]},
        "state":       {"dtype":"float32","shape":(8,), "names":["state"]},
        "actions":     {"dtype":"float32","shape":(7,), "names":["actions"]},
    }, ...
)
for episode in raw:
    for step in episode["steps"]:
        dataset.add_frame({"image": ..., "wrist_image": ..., "state": ..., "actions": ..., "task": "..."})
    dataset.save_episode()
```

Per-frame contract (matches `LiberoInputs`):
- `state`: 8-D = `[eef_pos(3), axis_angle(3), gripper_qpos[:2](2)]`
- `actions`: 7-D OSC_POSE = `[Δpos(3), Δaxis_angle(3), gripper(1)]`
- `image`, `wrist_image`: 256×256×3 uint8 (the policy resizes to 224 itself; stored at 256)
- `task`: language instruction string

Training config: `pi05_libero` in `config.py` uses `LeRobotLiberoDataConfig(repo_id="physical-intelligence/libero", extra_delta_transform=False)`. To fine-tune on our own dataset we override the data `repo_id` to point at our LeRobot dataset directory under `$HF_LEROBOT_HOME` — no converter changes required.

**Key insight**: there is no need to land in RLDS first or write HDF5 at all. The cleanest path is `env.step()` → directly `dataset.add_frame()` from inside the collection script.

## 5. Minimum implementation plan (4 small files, no LIBERO/openpi core changes — except blocker #1)

1. **Fix `_check_success()` for `trash_can` and `shelf`** in `LIBERO/libero/libero/envs/composuite/composuite_env.py` (~L478) (BLOCKER, see §7). ~12 LOC, reuses existing `goal_pos` / body positions; this is a justified core edit because demos cannot be filtered to "successful only" without it and the policy can't be evaluated post-fine-tune either. Keep the diff to a single function.
2. **`experiments/composuite_collect.py`** (~200 LOC). Reuses `_make_env` from `composuite_smoke.py`. State machine: `APPROACH_XY → DESCEND → CLOSE → LIFT → MOVE_XY → DESCEND_TARGET → OPEN → RETRACT`, generating 7-D OSC_POSE deltas from `obs["object_pos"]` and `obs["goal_pos"]` (offset z by +0.15 above; close threshold tuned from `obj_to_eef`). Per task, attempt N rollouts, keep only those with `info["success"]`.
3. **In the same script**, write directly to a `LeRobotDataset` named `composuite_box_pilot` (8-D state, 7-D actions, 256×256 images, fps=20). No HDF5 intermediate.
4. **Fine-tune config override**: a tiny new entry in `config.py` cloning `pi05_libero` with `repo_id="composuite_box_pilot"` and a smaller `num_train_steps` (e.g. 2k) — or simpler, pass `--data.repo_id` on the command line if the existing config supports it (verify before adding a new entry).

## 6. Estimated commands for collecting 5–10 demos per task

```bash
# Pilot — 5 successful demos per task, scripted, ~1 min total
source ~/miniconda3/etc/profile.d/conda.sh && conda activate libero
cd /home/amy/Projects/openpi
export MUJOCO_GL=egl HF_LEROBOT_HOME=$HOME/.cache/lerobot
python experiments/composuite_collect.py \
  --tasks Panda,box,none,pick_and_place \
          Panda,box,none,trash_can \
          Panda,box,none,shelf \
  --target-successes-per-task 5 \
  --max-attempts-per-task 30 \
  --max-steps 200 \
  --repo-id composuite_box_pilot \
  --video-out-path data/composuite/pilot_videos

# Sanity-check the dataset
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
  d=LeRobotDataset('composuite_box_pilot'); print(d.num_episodes, d.num_frames, d.features)"

# Scale-up after pilot passes
python experiments/composuite_collect.py ... --target-successes-per-task 30 \
  --repo-id composuite_box_v1
```

For fine-tuning (no commands run yet — gating step):
```bash
# In openpi .venv on GPU 1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=1 \
uv run scripts/train.py pi05_libero_composuite_pilot \
  --resume-from-checkpoint /home/amy/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --num-train-steps 2000 --batch-size 32
```
(exact command depends on what we name the new config entry in step 4 of §5; do not run until §7 blockers are cleared)

## 7. Blockers

1. **🔴 `_check_success()` only handles `pick_and_place` and `push`** — confirmed at `composuite_env.py` L478: no branch for `trash_can` / `shelf`, returns `False`. Without this fix we cannot filter demos for success and we cannot evaluate the fine-tuned policy on those tasks. **Fix is a small justified edit to one function**; matches the existing reward-function thresholds in `rewards.py`.
2. **🟡 Camera resolution**: the env currently renders at 256×256 (good — that's exactly what the converter expects); no change needed, but confirm in the collection script that `camera_heights=256, camera_widths=256` (default) is left intact.
3. **🟡 LeRobot version**: openpi's `pi05_libero` was trained with whatever LeRobot is pinned in `openpi/pyproject.toml`; the `libero` conda env may have a different version. Solution: do collection in the openpi `.venv` (same env as the policy server / training) instead of the `libero` conda env. The libero+robosuite stack is also present in openpi's `.venv` for the example to run. Verify before scaling.
4. **🟡 Push-objective early-termination wrapper**: `step()` early-terminates "illegal lifts" for `push` (irrelevant to our 3 tasks but worth knowing).
5. **🟢 No need to install upstream `composuite` package** — confirmed not used.
6. **🟢 No HDF5 ↔ RLDS adapter required** — direct `LeRobotDataset.add_frame` per env step is the canonical path.

## 8. Can fine-tuning run with existing OpenPI tools?

**Yes, with no converter required.** The full pipeline is already in openpi:
- Data: `LeRobotDataset` → consumed by `LeRobotLiberoDataConfig` (`config.py`)
- Training entrypoint: `scripts/train.py` with the existing `pi05_libero` config; only override is `repo_id` (and probably `num_train_steps` + `lr` for fine-tuning).
- Serving: `scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=<our_run>` — same call we already use for the base checkpoint.
- Eval: re-run `composuite_smoke.py` on the same 3 tasks against the new server.

The only **new** code is the collector (one file under `experiments/`) and the **only justified core edit** is finishing `_check_success()` for `trash_can` and `shelf`. Everything else is wrapper-only and config-only, consistent with the discipline we've kept through phases 2A–2D.

---

**Recommendation: proceed to step 1 (success-check fix) and step 2 (waypoint collector). Stop after collecting 5 demos per task and inspect the resulting LeRobot dataset + the success videos before scaling to 25–50 or starting any fine-tuning run.**
