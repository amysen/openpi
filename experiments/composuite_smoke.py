"""
Pi0.5 zero-shot evaluation on LIBERO-CompoSuite port.

Two CLI modes:
  1. Slice mode:  --task-specs "Panda,box,none,push" "Panda,box,none,shelf" ...
  2. Index mode:  --task-suite=composuite_full --task-indices 0 1 2

Optional --prompt-override forces the prompt (for language probes).
Writes per-episode JSON line records to --results-jsonl.
"""
import argparse
import collections
import hashlib
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from libero.libero import benchmark as libero_benchmark
from libero.libero.envs.composuite import CompoSuiteEnv
from robosuite.controllers import load_controller_config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
PRIMARY_TARGET_COLORS = ("red", "blue", "green")


def _target_color_for_seed(seed):
    return PRIMARY_TARGET_COLORS[int(seed) % len(PRIMARY_TARGET_COLORS)]


def _stable_seed(base_seed, *parts):
    key = "|".join(str(part) for part in parts).encode("utf-8")
    offset = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100000
    return int(base_seed) + offset


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _make_env(robot, object_type, obstacle, objective, seed, clutter="none", target_color="red"):
    cfg = load_controller_config(default_controller="OSC_POSE")
    env = CompoSuiteEnv(
        robot=robot,
        object_type=object_type,
        obstacle=obstacle,
        objective=objective,
        clutter=clutter,
        target_color=target_color,
        controller_configs=cfg,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=RES,
        camera_widths=RES,
    )
    env.seed(seed)
    return env


def _generate_language(robot, obj, obstacle, objective, clutter="none", target_color="red"):
    obj_name = obj.replace("_", " ")
    colored_obj = f"{target_color} {obj_name}".strip()
    if objective == "pick_and_place":
        action = f"pick up the {colored_obj} and place it in the wooden tray"
    elif objective == "push":
        action = f"push the {colored_obj} to the target bin"
    elif objective == "trash_can":
        action = f"pick up the {colored_obj} and drop it in the trash can"
    elif objective == "shelf":
        action = f"pick up the {colored_obj} and place it on the shelf"
    else:
        action = f"manipulate the {colored_obj}"
    if obstacle != "none":
        action = f"{action} while avoiding the {obstacle.replace('_', ' ')}"
    if clutter == "all_objects":
        action = f"{action} (other objects are present as distractors)"
    elif clutter == "decoys":
        decoy_colors = [color for color in PRIMARY_TARGET_COLORS if color != target_color]
        decoy_text = " and ".join(decoy_colors)
        action = (
            f"{action} ({decoy_text} {obj_name} decoys are also on the "
            f"table; ignore them and pick only the {colored_obj})"
        )
    elif clutter == "random":
        action = f"{action} (random clutter objects are present as distractors)"
    return action


def run_episode(env, client, prompt, args):
    env.reset()
    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    action_plan = collections.deque()
    replay_images = []
    cum_reward = 0.0
    done = False
    t = 0
    while t < args.max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
            replay_images.append(img)
            if not action_plan:
                # Force gripper_qpos to exactly 2 dims for non-Panda robots whose grippers
                # report >2 joints (e.g. Robotiq85=12). pi05_libero norm-stats expect 8-D state.
                gq = np.asarray(obs["robot0_gripper_qpos"]).flatten()
                if gq.size >= 2:
                    gq = gq[:2]
                else:
                    gq = np.concatenate([gq, np.zeros(2 - gq.size)])
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), gq)
                    ),
                    "prompt": str(prompt),
                }
                action_chunk = client.infer(element)["actions"]
                action_plan.extend(action_chunk[: args.replan_steps])
            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            cum_reward += float(reward)
            if done:
                break
            t += 1
        except Exception as e:
            logging.exception(f"step error: {e}")
            break
    diag = {}
    try:
        diag = {
            "obj_to_goal": float(np.linalg.norm(obs.get("object_to_goal_pos", np.zeros(3)))),
            "obj_to_eef": float(np.linalg.norm(obs.get("object_to_eef_pos", np.zeros(3)))),
            "obj_z": float(obs.get("object_pos", np.zeros(3))[2]),
        }
    except Exception:
        pass
    return done, t, cum_reward, replay_images, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--task-specs", nargs="+", default=None,
                   help="List of 'robot,object,obstacle,objective[,clutter]' strings. "
                        "The 5th `clutter` field is optional and defaults to 'none'.")
    p.add_argument("--task-suite", default=None)
    p.add_argument("--task-indices", type=int, nargs="+", default=None)
    p.add_argument("--prompt-override", default=None)
    p.add_argument("--prompt-tag", default=None, help="short tag for video filename when overriding")
    p.add_argument("--num-trials-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--video-out-path", default="data/composuite/videos")
    p.add_argument("--results-jsonl", default=None)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    np.random.seed(args.seed)

    tasks = []
    if args.task_specs:
        for s in args.task_specs:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) == 4:
                r, o, ob, gj = parts
                cl = "none"
            elif len(parts) == 5:
                r, o, ob, gj, cl = parts
            else:
                raise SystemExit(
                    f"Bad task spec {s!r}: expected 4 or 5 comma-separated fields"
                )
            tasks.append((r, o, ob, gj, cl))
    elif args.task_suite and args.task_indices:
        suite = libero_benchmark.get_benchmark_dict()[args.task_suite]()
        for i in args.task_indices:
            t = suite.get_task(i)
            tasks.append((t.robot, t.object_type, t.obstacle, t.objective,
                          getattr(t, "clutter", "none")))
    else:
        raise SystemExit("Need either --task-specs or --task-suite+--task-indices")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if args.results_jsonl:
        pathlib.Path(args.results_jsonl).parent.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    results = []
    t0_total = time.time()
    for (robot, obj, obstacle, objective, clutter) in tasks:
        target_color = _target_color_for_seed(args.seed)
        task_seed = _stable_seed(args.seed, robot, obj, obstacle, objective, clutter)
        gen_lang = _generate_language(
            robot, obj, obstacle, objective, clutter, target_color
        )
        prompt = args.prompt_override if args.prompt_override else gen_lang
        name = f"{robot}_{obj}_{obstacle}_{objective}"
        if clutter != "none":
            name = f"{name}_{clutter}"
        if args.prompt_tag:
            name = f"{name}__prompt-{args.prompt_tag}"
        logging.info(f"=== {name} | prompt={prompt!r} ===")
        env = _make_env(
            robot,
            obj,
            obstacle,
            objective,
            args.seed,
            clutter=clutter,
            target_color=target_color,
        )
        for ep in range(args.num_trials_per_task):
            episode_seed = task_seed + ep
            env.seed(episode_seed)
            t0 = time.time()
            done, steps, ret, frames, diag = run_episode(env, client, prompt, args)
            dt = time.time() - t0
            suffix = "success" if done else "failure"
            vp = pathlib.Path(args.video_out_path) / f"rollout_{name}_ep{ep}_{suffix}.mp4"
            if frames:
                imageio.mimwrite(vp, [np.asarray(x) for x in frames], fps=10)
            rec = {
                "task": name, "robot": robot, "object": obj, "obstacle": obstacle,
                "objective": objective, "clutter": clutter,
                "target_color": target_color,
                "prompt": prompt, "generated_language": gen_lang,
                "episode": ep, "seed": int(episode_seed),
                "success": bool(done), "steps": int(steps),
                "return": ret, "wall_seconds": round(dt, 1),
                "diag": diag, "video": str(vp),
            }
            results.append(rec)
            logging.info(f"  ep{ep}: success={done} steps={steps} return={ret:.2f} time={dt:.1f}s diag={diag}")
            if args.results_jsonl:
                with open(args.results_jsonl, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        if hasattr(env, "close"):
            try: env.close()
            except Exception: pass

    total_dt = time.time() - t0_total
    logging.info(f"=== DONE ({total_dt:.1f}s, {len(results)} eps) ===")
    succ = sum(r["success"] for r in results)
    logging.info(f"Total: {succ}/{len(results)}")


if __name__ == "__main__":
    main()
