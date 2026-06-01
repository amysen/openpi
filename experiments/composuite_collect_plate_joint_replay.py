"""Executable plate shelf-transfer demos from the successful preview motion.

The old shoulder-preview collector produced useful-looking videos by directly
editing simulator state after the plate was pulled off the first stand. Those
frames are not trainable: a policy cannot replay object teleports or unrecorded
joint edits. This collector uses that preview only as an internal joint-space
planner, then starts a fresh JOINT_POSITION environment with the same seed and
records only frames produced by env.step(action).
"""

import argparse
import contextlib
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np

import composuite_collect as cc


@contextlib.contextmanager
def _temporary_env(overrides):
    old = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _plan_joint_reference(robot, obj, obstacle, objective, seed, max_steps):
    joint_targets = []
    with _temporary_env(
        {
            "COMPOSUITE_PLATE_SHOULDER_ONLY_PREVIEW": "1",
            "COMPOSUITE_ALLOW_NONEXECUTABLE_DEMOS": "1",
        }
    ):
        env = cc._make_env(robot, obj, obstacle, objective, seed, controller="OSC_POSE")
        env.seed(seed)
        env.reset()
        ok, steps, _imgs, _wrs, _sts, acts, diag = cc.collect_episode(
            env, objective, max_steps, joint_targets=joint_targets
        )
        try:
            env.close()
        except Exception:
            pass
    if not ok:
        raise RuntimeError(f"preview planner failed: steps={steps} diag={diag}")
    if len(joint_targets) != len(acts):
        raise RuntimeError(f"planner produced {len(joint_targets)} joint targets for {len(acts)} actions")
    return np.asarray(joint_targets, dtype=np.float32), np.asarray(acts, dtype=np.float32)[:, -1], diag


def _execute_joint_reference(robot, obj, obstacle, objective, seed, joint_targets, gripper_cmds, max_extra_steps):
    env = cc._make_env(robot, obj, obstacle, objective, seed, controller="JOINT_POSITION")
    env.seed(seed)
    env.reset()

    dummy = np.zeros(int(env.action_dim), dtype=np.float32)
    dummy[-1] = cc.GRIP_OPEN
    obs, _, _, _ = env.step(dummy)
    for _ in range(10):
        obs, _, _, _ = env.step(dummy)

    images, wrists, states, actions = [], [], [], []
    total_steps = 0
    action_scale = float(getattr(env, "composuite_joint_action_scale", 0.12))
    max_extra_steps = max(1, int(max_extra_steps))
    last_grip = float(cc.GRIP_OPEN)
    saw_plate_grasp_close = False
    release_active = False
    release_steps = 0
    stopped_after_clearance = False

    for q_target, grip in zip(joint_targets, gripper_cmds):
        q_target = np.asarray(q_target, dtype=float)
        q_err = q_target - cc._robot_qpos(env).astype(float)
        max_abs_err = float(np.max(np.abs(q_err)))
        # Keep the motion visually and dynamically continuous. This matters
        # most after release, where the preview retreat can otherwise create a
        # big joint-space discontinuity that looks like a gripper snap.
        max_joint_step = 0.035 if float(grip) >= 0.0 else 0.020
        if last_grip >= 0.0 and float(grip) < 0.0:
            max_joint_step = 0.015
        reps = max(1, int(np.ceil(max_abs_err / max_joint_step)))
        reps = min(max_extra_steps, reps)

        for substep in range(reps):
            q_cur = cc._robot_qpos(env).astype(float)
            remaining = max(1, reps - substep)
            q_next = q_cur + (q_target - q_cur) / remaining
            dq = np.clip(q_next - q_cur, -action_scale, action_scale)
            act = np.zeros(int(env.action_dim), dtype=np.float32)
            act[: len(dq)] = np.clip(dq / action_scale, -1.0, 1.0)
            act[-1] = float(grip)

            img, wrist = cc._frames(obs)
            images.append(img)
            wrists.append(wrist)
            states.append(cc._state8(obs))
            actions.append(act)

            obs, _r, done, _info = env.step(act)
            total_steps += 1
            if float(grip) > 0.0:
                saw_plate_grasp_close = True
            if saw_plate_grasp_close and last_grip > 0.0 and float(grip) < 0.0:
                release_active = True
                release_steps = 0
            if release_active and float(grip) < 0.0:
                release_steps += 1
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=float)
                object_pos = np.array(obs["object_pos"], dtype=float)
                goal_pos = np.array(obs["goal_pos"], dtype=float)
                eef_clearance = float(np.linalg.norm(eef_pos[:2] - object_pos[:2]))
                object_settled = float(np.linalg.norm(object_pos - goal_pos)) < 0.06
                if object_settled and release_steps >= 12 and eef_clearance > 0.11:
                    stopped_after_clearance = True
                    break
            if done:
                break
        last_grip = float(grip)
        if stopped_after_clearance:
            break

    obj_to_goal = float(np.linalg.norm(np.array(obs["object_pos"]) - np.array(obs["goal_pos"])))
    ok = bool(env._check_success())
    diag = {
        "controller": "JOINT_POSITION",
        "frames_from_env_step_only": True,
        "obj_to_goal": obj_to_goal,
        "obj_z": float(obs["object_pos"][2]),
        "goal": np.round(np.array(obs["goal_pos"]), 4).tolist(),
        "object": np.round(np.array(obs["object_pos"]), 4).tolist(),
        "action_dim": int(env.action_dim),
        "joint_action_nonzero_frac": float(np.mean(np.linalg.norm(np.asarray(actions)[:, :7], axis=1) > 1e-6)),
        "release_steps_recorded": int(release_steps),
        "stopped_after_clearance": bool(stopped_after_clearance),
        "eef_clearance_after_release": float(
            np.linalg.norm(np.array(obs["robot0_eef_pos"])[:2] - np.array(obs["object_pos"])[:2])
        ),
    }
    try:
        env.close()
    except Exception:
        pass
    return ok, total_steps, images, wrists, states, actions, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--target-successes-per-task", type=int, default=5)
    parser.add_argument("--max-attempts-per-task", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-extra-steps-per-target", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = pathlib.Path(args.out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    summary = []

    for spec in args.tasks:
        robot, obj, obstacle, objective = [s.strip() for s in spec.split(",")]
        task_name = f"{robot}_{obj}_{obstacle}_{objective}"
        prompt = cc._generate_language(robot, obj, obstacle, objective)
        successes = 0
        for attempt in range(args.max_attempts_per_task):
            seed = args.seed + attempt
            t0 = time.time()
            try:
                refs, grips, plan_diag = _plan_joint_reference(robot, obj, obstacle, objective, seed, args.max_steps)
                ok, steps, imgs, wrs, sts, acts, diag = _execute_joint_reference(
                    robot,
                    obj,
                    obstacle,
                    objective,
                    seed,
                    refs,
                    grips,
                    args.max_extra_steps_per_target,
                )
            except Exception as exc:
                logging.exception("attempt %02d failed before video write: %s", attempt, exc)
                continue

            diag = {**diag, "planner_diag": plan_diag}
            tag = "success" if ok else "failure"
            vp = out / "videos" / f"{task_name}_ep{attempt:02d}_{tag}.mp4"
            imageio.mimwrite(vp, imgs, fps=20)
            logging.info("attempt %02d success=%s steps=%d diag=%s", attempt, ok, steps, diag)
            if not ok:
                continue

            ep_path = out / "episodes" / f"{task_name}_ep{successes:02d}.npz"
            np.savez_compressed(
                ep_path,
                image=np.stack(imgs).astype(np.uint8),
                wrist_image=np.stack(wrs).astype(np.uint8),
                state=np.stack(sts).astype(np.float32),
                actions=np.stack(acts).astype(np.float32),
                task=np.array(prompt),
                seed=np.array(seed, dtype=np.int64),
                task_spec=np.array(spec),
                controller=np.array("JOINT_POSITION"),
                goal_side=np.array(os.environ.get("COMPOSUITE_PLATE_GOAL_SIDE", "")),
            )
            summary.append(
                {
                    "task": task_name,
                    "ep": successes,
                    "attempt": attempt,
                    "seed": seed,
                    "steps": steps,
                    "wall_seconds": round(time.time() - t0, 1),
                    "video": str(vp),
                    "npz": str(ep_path),
                    "diag": diag,
                }
            )
            successes += 1
            if successes >= args.target_successes_per_task:
                break
        logging.info("=== %s: %d/%d ===", task_name, successes, args.target_successes_per_task)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote %d successful episodes to %s", len(summary), out)


if __name__ == "__main__":
    main()
