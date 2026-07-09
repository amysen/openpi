"""DAgger-style recovery-data collection for the plate-transfer task.

Rolls out a trained policy in CompoSuiteEnv until it stalls (or a step budget
runs out), then hands control to the scripted expert from composuite_collect,
entering the expert's phase pipeline at the point matching the current state.
If the expert completes the task, the TAKEOVER SEGMENT ONLY (states visited
from the drifted configuration + expert corrective actions) is saved as an NPZ
in the exact schema composuite_collect produces, so
convert_composuite_npz_to_lerobot.py ingests it unchanged.

Rationale: BC policies fail by compounding drift into states absent from the
demos. Expert takeover from policy-visited states yields exactly the missing
(off-distribution state -> corrective action) pairs. Failed takeovers (e.g.
the policy knocked the plate off its stand, which the side-grasp skill cannot
recover) are discarded by the same success filter the collector uses.

Run inside the openpi .venv on a GPU node (MUJOCO_GL=egl):

  uv run experiments/composuite_dagger.py \
      --config-name pi05_libero_composuite_transfer_v2 \
      --checkpoint-dir results/checkpoints/pi05_libero_composuite_transfer_v2/plate_doubleflip_blue_fixedlr/9000 \
      --out-dir data/composuite/plate_dagger_v1 \
      --num-successes 50
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import sys
import time

import imageio
import numpy as np

# libero comes from the checked-out fork; common.sh normally exports
# PYTHONPATH, but be defensive for direct invocations.
_LIBERO_DIR = os.environ.get(
    "LIBERO_DIR", "/home/pajak/compositional-learning-vla/repos/LIBERO"
)
if _LIBERO_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_DIR)

import composuite_collect as cc  # noqa: E402  (same directory)

PROMPT = "pull the plate off the stand and place it on top of the second stand"

# Geometry mirrored from cc._side_grasp_phases (horizontal_edge branch). Only
# the handful of constants needed to pick the expert's entry phase; the phase
# targets themselves come from cc._side_grasp_phases, not from these.
PLATE_GRIP_OFFSET = 0.035
SHELF2_DZ = 0.120            # destination-stand top above the table


def _quat2axisangle(quat):
    return cc._quat2axisangle(np.array(quat, dtype=np.float64).copy())


def _obs_to_policy_input(obs, prompt, convention="plate"):
    """Same conventions as eval_plate._obs_to_policy_input."""
    double_flip = convention in ("ootb", "vh_noresize")
    do_resize = convention in ("ootb", "v_resize")
    a, w = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
    img = np.ascontiguousarray(a[::-1, ::-1] if double_flip else a[::-1])
    wrist_img = np.ascontiguousarray(w[::-1, ::-1] if double_flip else w[::-1])
    if do_resize:
        from openpi_client import image_tools

        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
    gq = np.asarray(obs["robot0_gripper_qpos"]).flatten()
    gq = gq[:2] if gq.size >= 2 else np.concatenate([gq, np.zeros(2 - gq.size)])
    state = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(np.array(obs["robot0_eef_quat"])), gq)
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state.astype(np.float32),
        "prompt": str(prompt),
    }


def _infer_entry_phase(obs, env, spawn_obj_pos):
    """Decide where in the expert's phase pipeline to enter, or None if the
    state is unrecoverable for the side-grasp skill.

    Returns (entry_phase_name, plan_obs, reason):
      - plan_obs is the obs dict to build the phase plan from. For in-hand
        states we substitute the disc's SPAWN position so the plan's slide-off
        / carry geometry matches what the expert would have planned at t=0
        (the disc has moved with the gripper; planning off its current pose
        would re-derive a pull that is already partially done).
    """
    obj = np.array(obs["object_pos"], dtype=float)
    eef = np.array(obs["robot0_eef_pos"], dtype=float)
    goal = np.asarray(env.current_goal_pos, dtype=float)
    table_z = float(env.table_offset[2])
    gq = np.asarray(obs["robot0_gripper_qpos"]).flatten()[:2]
    spread = float(abs(gq[0]) + abs(gq[1]))

    if obj[2] < table_z + 0.05:
        return None, None, f"plate_fell(obj_z={obj[2]:.3f})"

    grasped = np.linalg.norm(eef - obj) < 0.09 and 0.006 < spread < 0.055

    if not grasped:
        # Re-grasp only works if the disc still sits on the source stand
        # (approximately where it spawned; the cantilever grasp needs it).
        moved = np.linalg.norm(obj[:2] - spawn_obj_pos[:2])
        at_rest_z = abs(obj[2] - spawn_obj_pos[2]) < 0.02
        if moved < 0.03 and at_rest_z:
            return "orient_wrist", obs, "regrasp_from_scratch"
        return None, None, f"not_grasped_and_displaced(dxy={moved:.3f})"

    # In hand: plan from the spawn pose, then route by how far along the
    # nominal trajectory the gripper already is.
    plan_obs = dict(obs)
    plan_obs["object_pos"] = spawn_obj_pos.copy()
    negative_y_goal = goal[1] < 0.0
    slide_off_dist = 0.26 if negative_y_goal else 0.22
    slide_off_eef_x = (spawn_obj_pos[0] - PLATE_GRIP_OFFSET) - slide_off_dist
    carry_clear_z = table_z + SHELF2_DZ + (0.130 if negative_y_goal else 0.060)
    if negative_y_goal:
        yaw = np.deg2rad(-35.0)
        place_dir = np.array([np.cos(yaw), np.sin(yaw)])
    else:
        place_dir = np.array([1.0, 0.0])
    goal_eef_xy = goal[:2] - place_dir * PLATE_GRIP_OFFSET

    if eef[0] > slide_off_eef_x + 0.03:
        return "slide_off", plan_obs, "inhand_over_stand"
    if eef[2] < carry_clear_z - 0.02:
        return "lift_carry", plan_obs, "inhand_low"
    if abs(eef[1] - goal_eef_xy[1]) > 0.03:
        return "traverse_y", plan_obs, "inhand_pre_traverse"
    if abs(eef[0] - goal_eef_xy[0]) > 0.025:
        return "align_x", plan_obs, "inhand_pre_align"
    return "settle_over_goal", plan_obs, "inhand_over_goal"


def _run_expert_phases(env, obs, phases, rec, max_steps, video_frames):
    """Step the expert through `phases`, recording (obs, action) into rec.
    Mirrors the stepping loop in cc.collect_episode (OSC_POSE only)."""
    success_seen = False
    steps = 0
    for ph in phases:
        if len(ph) == 7:
            name, target, target_quat, grip, tol, cap, pos_max = ph
        else:
            name, target, target_quat, grip, tol, cap = ph
            pos_max = 1.0
        for _ in range(cap):
            if steps >= max_steps:
                return obs, success_seen, steps, name
            eef = np.array(obs["robot0_eef_pos"])
            eef_quat = np.array(obs["robot0_eef_quat"])
            cur_target = np.asarray(target(obs)) if callable(target) else target
            if cur_target is None and target_quat is None:
                act = np.zeros(int(env.action_dim), dtype=np.float32)
                act[-1] = grip
            else:
                act = cc._go_to_pose(cur_target, target_quat, grip, eef, eef_quat, pos_max=pos_max)
            img, wrist = cc._frames(obs)
            rec["image"].append(img)
            rec["wrist_image"].append(wrist)
            rec["state"].append(cc._state8(obs))
            rec["actions"].append(np.asarray(act, dtype=np.float32))
            rec["phase"].append(name)
            video_frames.append(img)
            try:
                obs, _r, done, info = env.step(act)
            except ValueError:
                return obs, True, steps, name
            steps += 1
            if info.get("success"):
                success_seen = True
            if tol is not None:
                cur_eef = np.array(obs["robot0_eef_pos"])
                if cur_target is not None and np.linalg.norm(cur_target - cur_eef) < tol:
                    if target_quat is None:
                        break
                    if np.linalg.norm(cc._ori_error_axisangle(
                            np.array(obs["robot0_eef_quat"]), target_quat)) < 0.15:
                        break
                elif cur_target is None and target_quat is not None:
                    if np.linalg.norm(cc._ori_error_axisangle(
                            np.array(obs["robot0_eef_quat"]), target_quat)) < 0.10:
                        break
            if done:
                return obs, success_seen, steps, name
    return obs, success_seen, steps, "done"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-successes", type=int, default=50,
                   help="Stop after saving this many successful takeover episodes.")
    p.add_argument("--max-attempts", type=int, default=200)
    p.add_argument("--seed", type=int, default=21000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--max-policy-steps", type=int, default=350,
                   help="Force takeover after this many policy steps even without a stall.")
    p.add_argument("--min-policy-steps", type=int, default=30,
                   help="Give the policy at least this many steps before stall detection.")
    p.add_argument("--stall-steps", type=int, default=50,
                   help="Stall = eef net displacement below --stall-eef-eps over this window.")
    p.add_argument("--stall-eef-eps", type=float, default=0.01)
    p.add_argument("--max-expert-steps", type=int, default=900)
    p.add_argument("--target-color", default="blue",
                   choices=["random", "red", "blue", "green"])
    p.add_argument("--plate-thickness", default="thin", choices=["thin", "thick"],
                   help="Match the plate variant the checkpoint was trained on "
                        "(curriculum stage-1 uses 'thick').")
    p.add_argument("--init-pose-jitter", type=float, default=0.0,
                   help="If >0, perturb each robot joint by uniform[-J,+J] rad after "
                        "reset (before the policy rollout) so takeovers start from a "
                        "wider initial-state distribution.")
    p.add_argument("--obs-convention", default="plate",
                   choices=["plate", "ootb", "vh_noresize", "v_resize"])
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--skip-policy-success", action="store_true", default=True,
                   help="Do not save episodes the policy completes on its own (no expert data).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = pathlib.Path(args.out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    logging.info("Loading policy from %s ...", args.checkpoint_dir)
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config

    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, default_prompt=args.prompt)

    env = cc._make_env("Panda", "plate", "none", "pick_and_place", args.seed,
                       target_color=args.target_color,
                       plate_thickness=args.plate_thickness)
    task_name = "Panda_plate_none_pick_and_place"

    summary = []
    n_saved = 0
    n_policy_success = 0
    for attempt in range(args.max_attempts):
        if n_saved >= args.num_successes:
            break
        seed = args.seed + attempt
        env.seed(seed)
        obs = env.reset()
        try:
            import jax
            policy._rng = jax.random.key(seed)
        except Exception:
            pass
        if args.init_pose_jitter > 0.0:
            rng = np.random.RandomState(seed + 10_000)
            cc._perturb_joint_qpos(env, args.init_pose_jitter, rng)
            settle = np.zeros(int(env.action_dim), dtype=np.float32)
            settle[-1] = cc.GRIP_OPEN
            for _ in range(10):
                obs, _r, _d, _i = env.step(settle)
        spawn_obj_pos = np.array(obs["object_pos"], dtype=float).copy()

        # ---- policy rollout until stall / budget / success ----
        t0 = time.time()
        action_plan = collections.deque()
        eef_hist = collections.deque(maxlen=args.stall_steps)
        video_frames = []
        policy_success = False
        stalled_reason = "budget"
        pol_steps = 0
        for t in range(args.max_policy_steps):
            inp = _obs_to_policy_input(obs, args.prompt, args.obs_convention)
            if not action_plan:
                chunk = policy.infer(inp)["actions"]
                action_plan.extend(chunk[: args.replan_steps])
            act = np.asarray(action_plan.popleft(), dtype=np.float32)
            video_frames.append(cc._frames(obs)[0])
            try:
                obs, _r, done, info = env.step(act.tolist())
            except Exception as e:
                logging.warning("env.step error during policy rollout: %s", e)
                break
            pol_steps = t + 1
            if info.get("success"):
                policy_success = True
                break
            obj_z = float(np.array(obs["object_pos"])[2])
            if obj_z < float(env.table_offset[2]) - 0.03:
                stalled_reason = "plate_on_floor"
                break
            eef_hist.append(np.array(obs["robot0_eef_pos"], dtype=float))
            if t >= args.min_policy_steps and len(eef_hist) == args.stall_steps:
                disp = np.linalg.norm(eef_hist[-1] - eef_hist[0])
                if disp < args.stall_eef_eps:
                    stalled_reason = "stall"
                    break

        if policy_success:
            n_policy_success += 1
            logging.info("attempt %03d: policy succeeded on its own (%d steps) — not saved",
                         attempt, pol_steps)
            summary.append({"attempt": attempt, "seed": seed, "outcome": "policy_success",
                            "policy_steps": pol_steps})
            continue

        # ---- expert takeover ----
        entry, plan_obs, reason = _infer_entry_phase(obs, env, spawn_obj_pos)
        if entry is None:
            logging.info("attempt %03d: unrecoverable after %d policy steps (%s)",
                         attempt, pol_steps, reason)
            summary.append({"attempt": attempt, "seed": seed, "outcome": "unrecoverable",
                            "policy_steps": pol_steps, "reason": reason})
            continue

        phases = cc._side_grasp_phases(plan_obs, env, "pick_and_place", "horizontal_edge")
        names = [ph[0] for ph in phases]
        if entry not in names:
            logging.warning("attempt %03d: entry phase %r not in plan %s", attempt, entry, names)
            continue
        phases = phases[names.index(entry):]

        rec = {"image": [], "wrist_image": [], "state": [], "actions": [], "phase": []}
        obs, success_seen, exp_steps, last_phase = _run_expert_phases(
            env, obs, phases, rec, args.max_expert_steps, video_frames)

        # Settle briefly, then final success check (mirrors the collector).
        hold = np.zeros(int(env.action_dim), dtype=np.float32)
        hold[-1] = cc.GRIP_OPEN
        for _ in range(15):
            try:
                obs, _r, _d, info = env.step(hold)
                if info.get("success"):
                    success_seen = True
            except Exception:
                break
        try:
            ok = bool(success_seen or env._check_success())
        except Exception:
            ok = success_seen

        dt = time.time() - t0
        tag = "success" if ok else "failure"
        vp = out / "videos" / f"{task_name}_dagger_ep{attempt:03d}_{tag}.mp4"
        try:
            imageio.mimwrite(vp, video_frames, fps=20)
        except Exception as e:
            logging.warning("video write failed: %s", e)

        logging.info(
            "attempt %03d: %s | policy=%d steps (%s) -> entry=%s (%s) | expert=%d steps "
            "(last=%s) | %.1fs",
            attempt, tag, pol_steps, stalled_reason, entry, reason, exp_steps, last_phase, dt)

        if ok and len(rec["state"]) > 10:
            ep_path = out / "episodes" / f"{task_name}_dagger_ep{n_saved:02d}.npz"
            np.savez_compressed(
                ep_path,
                image=np.stack(rec["image"]).astype(np.uint8),
                wrist_image=np.stack(rec["wrist_image"]).astype(np.uint8),
                state=np.stack(rec["state"]).astype(np.float32),
                actions=np.stack(rec["actions"]).astype(np.float32),
                task=np.array(args.prompt),
                seed=np.array(seed, dtype=np.int64),
                task_spec=np.array("Panda,plate,none,pick_and_place"),
                controller=np.array("OSC_POSE"),
                phase=np.array(rec["phase"]),
                plate_thickness=np.array(args.plate_thickness),
                init_pose_jitter=np.array(args.init_pose_jitter, dtype=np.float32),
                dagger_entry_phase=np.array(entry),
                dagger_policy_steps=np.array(pol_steps, dtype=np.int64),
                dagger_checkpoint=np.array(str(args.checkpoint_dir)),
            )
            n_saved += 1
            summary.append({"attempt": attempt, "seed": seed, "outcome": "saved",
                            "entry_phase": entry, "entry_reason": reason,
                            "policy_steps": pol_steps, "stall_reason": stalled_reason,
                            "expert_steps": exp_steps, "npz": str(ep_path),
                            "video": str(vp)})
        else:
            summary.append({"attempt": attempt, "seed": seed, "outcome": "takeover_failed",
                            "entry_phase": entry, "policy_steps": pol_steps,
                            "expert_steps": exp_steps, "last_phase": last_phase,
                            "video": str(vp)})

    with open(out / "summary.json", "w") as f:
        json.dump({"saved": n_saved, "policy_successes": n_policy_success,
                   "attempts": len(summary), "episodes": summary}, f, indent=2)
    logging.info("Done: %d takeover episodes saved (%d policy-only successes) -> %s",
                 n_saved, n_policy_success, out)


if __name__ == "__main__":
    main()
