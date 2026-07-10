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
import os
import pathlib
import sys
import time

# This script is spawned by the post-save eval hook WHILE training holds the
# GPU (train_pi05_libero.sh runs training at XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# and subprocess env is inherited, so without an override the eval process
# tries to preallocate the same 90% and dies with RESOURCE_EXHAUSTED). Force
# modest on-demand allocation for THIS process; must happen before jax import.
# Preferred setup: the training job allocates 2 GPUs and sets
# OPENPI_EVAL_CUDA_DEVICE so this process runs on its own card (see
# train_sanity.sh) -- then no memory partitioning is needed at all.
if os.environ.get("OPENPI_EVAL_CUDA_DEVICE"):
    # After narrowing CUDA_VISIBLE_DEVICES, this process sees exactly one GPU
    # (index 0 in its view), and EGL enumerates the same visible set -- so no
    # MUJOCO_EGL_DEVICE_ID override is needed (or correct) here.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OPENPI_EVAL_CUDA_DEVICE"]
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("OPENPI_EVAL_MEM_FRACTION", "0.3")

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


def _build_env(
    seed: int, clutter: str = "none", target_color: str = "blue",
    objective: str = "pick_and_place", object_type: str = "plate",
    plate_thickness: str = "thin",
):
    from libero.libero.envs.composuite import CompoSuiteEnv
    from robosuite.controllers import load_controller_config

    cfg = load_controller_config(default_controller="OSC_POSE")
    env = CompoSuiteEnv(
        robot="Panda",
        object_type=object_type,  # plate (train) | box | dumbbell | hollow_box ...
        obstacle="none",
        objective=objective,  # pick_and_place (train) | push | trash_can | shelf
        clutter=clutter,  # "none" (train dist.) | "all_objects" | "decoys" | "random"
        # target_color defaults to "random", which draws a color at construction
        # NOT pinned to the seed -> varies per process. Pin it (e.g. "green") for a
        # controlled comparison where only `clutter` changes between runs.
        target_color=target_color,
        plate_thickness=plate_thickness,
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


def _obs_to_policy_input(obs, prompt, convention: str = "ootb"):
    # Image convention, with flip and resize controlled INDEPENDENTLY so we can
    # isolate which one matters:
    #   plate        = [::-1]        , no resize   (the plate fine-tuning dataset)
    #   ootb         = [::-1, ::-1]  , resize_pad  (openpi LIBERO eval convention)
    #   vh_noresize  = [::-1, ::-1]  , no resize   (flip only)
    #   v_resize     = [::-1]        , resize_pad  (resize only)
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


def _run_trial(
    env, policy, max_steps: int, replan_steps: int, seed: int, prompt: str = PROMPT,
    end_on_release: bool = False, obs_convention: str = "ootb",
    num_steps_wait: int = 10,
):
    env.seed(seed)
    obs = env.reset()
    # Reset the policy's flow-matching sampling RNG to a fixed per-seed key so each
    # (checkpoint, seed) rollout is reproducible. Without this the Policy advances
    # self._rng on every infer() and never resets it (policy.py), so a trial's noise
    # depends on how many infer calls ran in prior trials -> identical layouts can
    # flip success/failure with seed-list order. (No-op for non-JAX policies.)
    try:
        import jax

        policy._rng = jax.random.key(seed)
    except Exception:
        pass
    # After success fires, keep rolling so the video shows the real settle. Default:
    # a short fixed tail. With end_on_release: keep going until the model actually
    # OPENS the gripper (releases the object) -- success can fire while still grasped,
    # so a fixed tail may cut off before the policy's own release -- then stop after a
    # brief settle. This is the model's real behaviour; nothing is scripted.
    # Upstream-fidelity settle: openpi's LIBERO eval (examples/libero/main.py,
    # num_steps_wait=10) steps a dummy open-gripper action while objects
    # stabilize after reset, before the first inference.
    for _ in range(num_steps_wait):
        try:
            obs, _r, _done, _info = env.step([0.0] * 6 + [-1.0])
        except Exception as e:
            logging.warning(f"env.step error during settle: {e}")
            break
    POST_SUCCESS_FRAMES = 45
    RELEASE_SPREAD = 0.05       # finger-spread above this => gripper opened (released)
    SETTLE_AFTER_RELEASE = 50  # let the plate fall + come to rest before we read its position
    RELEASE_SAFETY_CAP = 250    # stop anyway if it never opens after success
    action_plan = collections.deque()
    frames = []
    last_success = False
    success_t = None
    released_t = None
    # Trajectory progress trackers: capture partial progress the final-state snapshot
    # misses (e.g. carried the plate close to the goal, then dropped it at the end).
    min_o2g = float("inf")     # closest the object ever got to the goal (xy)
    min_o2e = float("inf")     # closest the gripper ever got to the object (the approach/reach)
    max_obj_z = -float("inf")  # highest the object was lifted
    approach_quat = None       # gripper orientation (x,y,z,w) at the closest-approach frame
    max_wrist_tilt = 0.0       # max wrist rotation from vertical over the trajectory (deg)
    for t in range(max_steps):
        inp, img = _obs_to_policy_input(obs, prompt, obs_convention)
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
        try:
            _op = np.array(obs["object_pos"])
            _gp = np.asarray(env.current_goal_pos)
            min_o2g = min(min_o2g, float(np.linalg.norm(_op[:2] - _gp[:2])))
            max_obj_z = max(max_obj_z, float(_op[2]))
            _ee = np.array(obs["robot0_eef_pos"])
            _q = np.asarray(obs["robot0_eef_quat"]).ravel()[:4]  # (x,y,z,w)
            # wrist tilt from vertical of the gripper approach axis: top-down grasp ~0deg,
            # side-grasp (wrist rotated horizontal) ~90deg.
            _zc = max(0.0, min(1.0, abs(1.0 - 2.0 * (float(_q[0]) ** 2 + float(_q[1]) ** 2))))
            _tilt = float(np.degrees(np.arccos(_zc)))
            max_wrist_tilt = max(max_wrist_tilt, _tilt)
            _d_o2e = float(np.linalg.norm(_ee - _op))
            if _d_o2e < min_o2e:
                min_o2e = _d_o2e
                approach_quat = [float(v) for v in _q]
        except Exception:
            pass
        if info.get("terminated_reason") == "illegal_lift":
            # Push objective: the env ends the episode when the object rises
            # >3cm off the table (PushReward). Honor it as a terminal failure,
            # else a pick-and-carry to the goal masquerades as push success
            # (the reason OOTB-sweep push rows were quarantined).
            break
        if info.get("success") and not last_success:
            last_success = True
            success_t = t
        if last_success:
            if end_on_release:
                gq = np.asarray(obs["robot0_gripper_qpos"]).flatten()
                spread = float(abs(gq[0]) + abs(gq[1])) if gq.size >= 2 else 1.0
                if released_t is None and spread > RELEASE_SPREAD:
                    released_t = t
                if released_t is not None and (t - released_t) >= SETTLE_AFTER_RELEASE:
                    break
                if (t - success_t) >= RELEASE_SAFETY_CAP:
                    break
            elif (t - success_t) >= POST_SUCCESS_FRAMES:
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
            "obj_to_goal_min": (None if min_o2g == float("inf") else min_o2g),
            "obj_to_eef_min": (None if min_o2e == float("inf") else min_o2e),
            "obj_z_max": (None if max_obj_z == -float("inf") else max_obj_z),
            # wrist orientation: tilt at the closest-approach frame + max over the rollout.
            # top-down (box-style) ~0deg; plate side-grasp ~90deg.
            "approach_quat": approach_quat,
            "wrist_tilt_at_approach_deg": (
                None if approach_quat is None
                else float(np.degrees(np.arccos(
                    max(0.0, min(1.0, abs(1.0 - 2.0 * (approach_quat[0] ** 2 + approach_quat[1] ** 2)))))))
            ),
            "wrist_tilt_max_deg": max_wrist_tilt,
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
    p.add_argument(
        "--clutter",
        default="none",
        choices=["none", "all_objects", "decoys", "random"],
        help="Background distractors. Training used 'none'; other modes are OOD and "
        "probe whether a clean-scene success survives added clutter (brittleness).",
    )
    p.add_argument(
        "--target-color",
        default="blue",
        choices=["random", "red", "blue", "green"],
        help="Plate color, default 'blue' to MATCH the pinned training data "
        "(composuite_collect.py --target-color blue). 'random' draws per-process and "
        "is NOT seed-pinned, so it can mismatch the training color -- avoid for eval.",
    )
    p.add_argument(
        "--objective",
        default="pick_and_place",
        choices=["pick_and_place", "push", "trash_can", "shelf"],
        help="Task goal. Training used 'pick_and_place'; other objectives are OOD "
        "compositional-reuse probes (same plate, different place-target).",
    )
    p.add_argument(
        "--object",
        dest="object_type",
        default="plate",
        help="Manipulated object (plate=train; box/dumbbell/... probe whether "
        "fine-tuning on the plate kept other-object skills / caused forgetting).",
    )
    p.add_argument(
        "--obs-convention",
        default="ootb",
        choices=["plate", "ootb", "vh_noresize", "v_resize"],
        help="Image convention. 'ootb' (default) = [::-1,::-1] + resize_with_pad(224) "
        "= pi0.5-LIBERO's pretraining/eval convention (openpi examples/libero/main.py) "
        "and the current collector's orientation; closed-loop camera_ab was decisive "
        "for it. 'plate' = [::-1], only for legacy single-flip-era checkpoints.",
    )
    p.add_argument(
        "--prompt",
        default=PROMPT,
        help="Language instruction. Defaults to the training prompt; override to "
        "match a new --objective (e.g. drop the plate in the trash can).",
    )
    p.add_argument(
        "--plate-thickness",
        default="thin",
        choices=["thin", "thick"],
        help="Plate curriculum variant. Evaluate stage-1 (thick-trained) checkpoints "
        "with 'thick'; final numbers should always be reported on 'thin'.",
    )
    p.add_argument(
        "--end-on-release",
        action="store_true",
        help="End each rollout shortly after the model actually opens the gripper "
        "(releases the object), instead of a fixed post-success tail. Success can "
        "fire while still grasped, so this captures the policy's real release.",
    )
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated explicit seed list (e.g. the recorded training-demo "
        "seeds). Overrides --seed/--num-trials so you can replay exact training "
        "layouts and check in-distribution success.",
    )
    args = p.parse_args()

    # Explicit seed list (training replay) takes precedence over --seed/--num-trials.
    seed_list = (
        [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        if args.seeds
        else [args.seed + i for i in range(args.num_trials)]
    )

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
        train_config, args.checkpoint_dir, default_prompt=args.prompt
    )

    env = _build_env(
        seed_list[0], clutter=args.clutter, target_color=args.target_color,
        objective=args.objective, object_type=args.object_type,
        plate_thickness=args.plate_thickness,
    )
    logging.info(f"object={args.object_type}  objective={args.objective}  prompt={args.prompt!r}")

    results = []
    for i, seed_i in enumerate(seed_list):
        t0 = time.time()
        success, steps, frames, diag = _run_trial(
            env, policy, args.max_steps, args.replan_steps, seed_i, prompt=args.prompt,
            end_on_release=args.end_on_release, obs_convention=args.obs_convention,
        )
        # For trash_can, the env's predicate fires while the plate is still held at
        # the opening (lenient + transient). Require the SETTLED plate to actually be
        # INSIDE the can: within the can footprint (xy) and dropped below the rim. Pair
        # with --end-on-release so the plate has settled after the gripper opens.
        if args.objective == "trash_can" and args.object_type == "plate":
            xy = diag.get("obj_to_goal")
            oz = diag.get("obj_z")
            gz = diag.get("goal_z")
            success = bool(
                xy is not None and oz is not None and gz is not None
                and xy < 0.06 and oz < gz + 0.10
            )
            diag["deposited_in_can"] = success
        dt = time.time() - t0
        tag = "success" if success else "failure"
        vp = out / f"step_{step_tag}_trial{i:02d}_seed{seed_i}_{tag}.mp4"
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
        "num_trials": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "trials": results,
    }
    with open(out / f"step_{step_tag}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Eval summary at step {step_tag}: success_rate={summary['success_rate']:.2f}")


if __name__ == "__main__":
    main()
