"""One-off visual sanity check for the new side-shelf placement.

Resets a CompoSuiteEnv with objective='shelf' for each of (box, dumbbell, plate)
using the Panda robot and no obstacle, and saves a single agentview PNG so we
can confirm:
  1. the shelf has moved to the side (y=+0.30, x=+0.05),
  2. the dumbbell and plate assets load and place correctly.

Usage (from /home/amy/Projects/openpi with .venv active):
  MUJOCO_GL=egl python experiments/render_shelf_check.py \
      --out-dir data/composuite/shelf_check
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as imageio
import numpy as np

from libero.libero.envs.composuite import CompoSuiteEnv
from robosuite.controllers import load_controller_config


def render_one(robot: str, object_type: str, out_dir: pathlib.Path) -> pathlib.Path:
    env = CompoSuiteEnv(
        robot=robot,
        object_type=object_type,
        obstacle="none",
        objective="shelf",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
        horizon=1,
        ignore_done=True,
    )
    env.seed(0)
    obs = env.reset()
    img = obs["agentview_image"]
    # robosuite returns the image upside down; flip to natural orientation.
    img = np.flipud(img)
    out_path = out_dir / f"shelf_{object_type}.png"
    imageio.imwrite(out_path, img)
    env.close()
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/composuite/shelf_check")
    p.add_argument("--robot", default="Panda")
    p.add_argument("--objects", nargs="+", default=["box", "dumbbell", "plate"])
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj in args.objects:
        path = render_one(args.robot, obj, out_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
