"""Evaluate a pi05_libero_composuite_transfer checkpoint by rolling out the
trained policy in the CompoSuite plate-pick-and-place env and saving MP4s.

Designed to be called from train.py after each save_interval so a user can
watch the VLA's skill grow over the course of training. Outputs:
  <out-dir>/step_<step>_trial<i>_<success|failure>.mp4
  <out-dir>/step_<step>_eval.json     (success rate, avg steps, etc.)

Usage:
  python experiments/eval_plate.py \
      --config-name pi05_libero_composuite_transfer \
      --checkpoint-dir checkpoints/pi05_libero_composuite_transfer/exp1/100 \
      --out-dir checkpoints/pi05_libero_composuite_transfer/exp1/eval_videos \
      --num-trials 3 \
      --max-steps 800 \
      --replan-steps 5 \
      --seed 7001
"""
import argparse
import collections
import json
import logging
import math
import pathlib
import sys
import time

import imageio
import numpy as np

# The libero package isn't on the openpi venv's default path; inject the
# checked-out repo so `libero.libero.envs.composuite` is importable.
_LIBERO_REPO = "/home/amy/Projects/LIBERO"
if _LIBERO_REPO not in sys.path:
    sys.path.insert(0, _LIBERO_REPO)

# Heavy imports gated under main() so the file imports cheaply.


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0, abs_tol=1e-8):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _build_env(seed: int):
    from libero.libero.envs.composuite import CompoSuiteEnv
    from robosuite.controllers import load_controller_config

    cfg = load_controller_config(default_controller="OSC_POSE")
    env = CompoSuiteEnv(
        robot="Panda",
        object_type="plate",
        obstacle="none",
        objective="pick_and_place",
        controller_configs=cfg,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        horizon=2000,
        ignore_done=True,
        use_composuite_agentview=False,  # LIBERO floor-manipulation camera
    )
    env.seed(seed)
    return env


def _obs_to_policy_input(obs, prompt):
    # Match the LIBERO image convention used in the training dataset:
    # vertical flip only (NOT [::-1, ::-1]).
    img = np.ascontiguousarray(obs["agentview_image"][::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    gq = np.asarray(obs["robot0_gripper_qpos"]).flatten()
    if gq.size >= 2:
        gq = gq[:2]
    else:
        gq = np.concatenate([gq, np.zeros(2 - gq.size)])
    state = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(np.array(obs["robot0_eef_quat"])), gq)
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state.astype(np.float32),
        "prompt": str(prompt),
    }, img


PROMPT = "pull the plate off the stand and place it on top of the second stand"


def _run_trial(env, policy, max_steps: int, replan_steps: int, seed: int):
    env.seed(seed)
    obs = env.reset()
    action_plan = collections.deque()
    frames = []
    last_success = False
    for t in range(max_steps):
        inp, img = _obs_to_policy_input(obs, PROMPT)
        frames.append(img)
        if not action_plan:
            chunk = policy.infer(inp)["actions"]
            action_plan.extend(chunk[:replan_steps])
        action = action_plan.popleft()
        try:
            obs, _r, _done, info = env.step(np.asarray(action).tolist())
        except Exception as e:
            logging.warning(f"env.step error: {e}")
            break
        if info.get("success"):
            last_success = True
            break
    # Diagnostic snapshot
    diag = {}
    try:
        obj_pos = np.array(obs["object_pos"])
        goal_pos = env.current_goal_pos.copy()
        diag = {
            "obj_to_goal": float(np.linalg.norm(obj_pos[:2] - goal_pos[:2])),
            "obj_z": float(obj_pos[2]),
            "goal_z": float(goal_pos[2]),
        }
    except Exception:
        pass
    return last_success, t + 1, frames, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-trials", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7001)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step number for filenames is the last path segment of checkpoint_dir
    # (openpi saves under <ckpt_base>/<config_name>/<step>/).
    step_tag = pathlib.Path(args.checkpoint_dir).name

    logging.info(f"Loading policy from {args.checkpoint_dir}...")
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config

    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, default_prompt=PROMPT
    )

    env = _build_env(args.seed)

    results = []
    for i in range(args.num_trials):
        seed_i = args.seed + i
        t0 = time.time()
        success, steps, frames, diag = _run_trial(
            env, policy, args.max_steps, args.replan_steps, seed_i
        )
        dt = time.time() - t0
        tag = "success" if success else "failure"
        vp = out / f"step_{step_tag}_trial{i:02d}_{tag}.mp4"
        try:
            imageio.mimwrite(vp, frames, fps=20)
        except Exception as e:
            logging.warning(f"video write failed: {e}")
        logging.info(f"  trial {i:02d}: success={success} steps={steps} time={dt:.1f}s diag={diag}")
        results.append(
            {
                "trial": i,
                "seed": seed_i,
                "success": success,
                "steps": steps,
                "wall_seconds": round(dt, 1),
                "diag": diag,
                "video": str(vp),
            }
        )

    summary = {
        "step": step_tag,
        "num_trials": args.num_trials,
        "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "trials": results,
    }
    with open(out / f"step_{step_tag}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Eval summary at step {step_tag}: success_rate={summary['success_rate']:.2f}")


if __name__ == "__main__":
    main()
