#!/usr/bin/env python
"""Replay a CompoSuite NPZ demo using only its recorded actions.

Use this as a gate before converting demos to LeRobot: if a saved episode
cannot be replayed from the same env seed with the recorded actions, the policy
cannot learn to reproduce the visible video from those actions either.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib

import imageio
import numpy as np
from robosuite.controllers import load_controller_config

from libero.libero.envs.composuite import CompoSuiteEnv


RES = 256
GRIP_OPEN = -1.0


def _make_env(robot, object_type, obstacle, objective, seed, controller):
    cfg = load_controller_config(default_controller=controller)
    if controller == "JOINT_POSITION":
        cfg["output_max"] = 0.12
        cfg["output_min"] = -0.12
        cfg["kp"] = 150
    env = CompoSuiteEnv(
        robot=robot,
        object_type=object_type,
        obstacle=obstacle,
        objective=objective,
        controller_configs=cfg,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=RES,
        camera_widths=RES,
        horizon=800,
        ignore_done=True,
    )
    env.seed(seed)
    env.composuite_controller = controller
    return env


def _frame(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _read_scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "ndim", 1) == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


@contextlib.contextmanager
def _temporary_env(name, value):
    old = os.environ.get(name)
    try:
        if value:
            os.environ[name] = str(value)
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Episode .npz written by composuite_collect.py")
    parser.add_argument("--task-spec", default=None, help="robot,object,obstacle,objective")
    parser.add_argument("--controller", default=None, help="Robosuite controller, defaults to npz metadata or OSC_POSE")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=11)
    parser.add_argument("--video-out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    path = pathlib.Path(args.npz)
    data = np.load(path, allow_pickle=True)
    task_spec = args.task_spec or _read_scalar(data, "task_spec")
    if task_spec is None:
        raise SystemExit("Need --task-spec because the npz does not contain task_spec")
    seed = args.seed if args.seed is not None else _read_scalar(data, "seed")
    if seed is None:
        raise SystemExit("Need --seed because the npz does not contain seed")
    robot, obj, obstacle, objective = [part.strip() for part in task_spec.split(",")]
    actions = data["actions"].astype(np.float32)
    controller = args.controller or _read_scalar(data, "controller", "OSC_POSE")
    goal_side = _read_scalar(data, "goal_side", os.environ.get("COMPOSUITE_PLATE_GOAL_SIDE", ""))

    with _temporary_env("COMPOSUITE_PLATE_GOAL_SIDE", goal_side):
        env = _make_env(robot, obj, obstacle, objective, int(seed), str(controller))
        env.reset()
        dummy_action = np.zeros(int(env.action_dim), dtype=np.float32)
        dummy_action[-1] = GRIP_OPEN
        obs = None
        for _ in range(args.warmup_steps):
            obs, _, _, _ = env.step(dummy_action)
        start_obj = np.array(obs["object_pos"], dtype=np.float32)
        frames = [_frame(obs)]
        rewards = []
        last_info = {}
        for action in actions:
            obs, reward, _done, last_info = env.step(action)
            rewards.append(float(reward))
            frames.append(_frame(obs))

    object_pos = np.array(obs["object_pos"], dtype=np.float32)
    goal_pos = np.array(obs["goal_pos"], dtype=np.float32)
    rec = {
        "npz": str(path),
        "task_spec": task_spec,
        "controller": str(controller),
        "goal_side": str(goal_side),
        "seed": int(seed),
        "num_actions": int(len(actions)),
        "success": bool(env._check_success()),
        "env_info_success": bool(last_info.get("success", False)),
        "obj_to_goal": float(np.linalg.norm(object_pos - goal_pos)),
        "object_displacement": float(np.linalg.norm(object_pos - start_obj)),
        "return": float(sum(rewards)),
    }
    if args.video_out:
        video_out = pathlib.Path(args.video_out)
        video_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_out, frames, fps=20)
        rec["video"] = str(video_out)
    text = json.dumps(rec, indent=2)
    if args.json_out:
        json_out = pathlib.Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n")
    print(text)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
