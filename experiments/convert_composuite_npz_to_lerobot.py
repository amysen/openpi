"""
Convert CompoSuite NPZ episodes (produced by experiments/composuite_collect.py)
into a LeRobotDataset compatible with the openpi LIBERO data pipeline.

Run inside the openpi .venv (lerobot requires Python >=3.10):

  source .venv/bin/activate
  python experiments/convert_composuite_npz_to_lerobot.py \
      --episodes-dir data/composuite/box_objective_v1/episodes \
      --repo-id composuite_box_objective_v1

Each input NPZ contains:
  image:       (T, 256, 256, 3) uint8
  wrist_image: (T, 256, 256, 3) uint8
  state:       (T, 8)           float32  [eef_pos(3), axisangle(3), gripper_qpos[:2]]
  actions:     (T, A)           float32  controller actions + gripper
  task:        scalar str       language prompt

Output dataset is written to $HF_LEROBOT_HOME/<repo_id>/.
"""

import argparse
import pathlib
import shutil
import sys
import time

import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-dir", required=True,
                   help="Directory containing *.npz episode files")
    p.add_argument("--repo-id", default="composuite_box_objective_v1")
    p.add_argument("--robot-type", default="panda")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--overwrite", action="store_true",
                   help="Delete the output directory if it already exists")
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--image-writer-processes", type=int, default=8)
    p.add_argument("--start-idx", type=int, default=0,
                   help="Index of first episode to process (inclusive). Use with --end-idx for shard-parallel conversion.")
    p.add_argument("--end-idx", type=int, default=None,
                   help="Index past the last episode to process (exclusive). Defaults to all episodes.")
    p.add_argument(
        "--trim-leading-idle",
        action="store_true",
        help="Drop the leading frames of each episode where the EE is essentially static "
             "(see --idle-action-thresh / --idle-min-run / --idle-max-trim).",
    )
    p.add_argument(
        "--idle-action-thresh", type=float, default=5e-3,
        help="Per-frame action-xyz norm below this counts as idle (OSC_POSE delta units).",
    )
    p.add_argument(
        "--idle-min-run", type=int, default=10,
        help="Minimum consecutive idle frames at the start before trimming kicks in.",
    )
    p.add_argument(
        "--idle-max-trim-frac", type=float, default=0.5,
        help="Never trim more than this fraction of the episode length.",
    )
    args = p.parse_args()

    ep_dir = pathlib.Path(args.episodes_dir)
    npz_paths = sorted(ep_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"No .npz files in {ep_dir}")
    end_idx = args.end_idx if args.end_idx is not None else len(npz_paths)
    npz_paths = npz_paths[args.start_idx:end_idx]
    if not npz_paths:
        raise SystemExit(f"No episodes in slice [{args.start_idx}:{end_idx}] (total {len(sorted(ep_dir.glob('*.npz')))}).")
    print(f"Processing episodes [{args.start_idx}:{end_idx}] ({len(npz_paths)} episodes)")

    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output dataset already exists at {out_path}. Pass --overwrite to replace.")
        shutil.rmtree(out_path)

    print(f"Creating LeRobotDataset at {out_path}")
    with np.load(npz_paths[0], allow_pickle=True) as first_ep:
        action_dim = int(first_ep["actions"].shape[-1])
    for pth in npz_paths[1:]:
        with np.load(pth, allow_pickle=True) as d:
            this_dim = int(d["actions"].shape[-1])
        if this_dim != action_dim:
            raise SystemExit(
                f"Mixed action dimensions are not supported: {npz_paths[0].name} has {action_dim}, "
                f"{pth.name} has {this_dim}. Keep OSC_POSE and IK_POSE demos in separate datasets."
            )
    print(f"  action_dim={action_dim}", flush=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    total_frames = 0
    start = time.time()
    grand_total_frames = 0
    # Pre-scan to know how many frames in total for ETA reporting
    print("Pre-scanning episode lengths...", flush=True)
    ep_lens = []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as d:
            ep_lens.append(int(d["state"].shape[0]))
    grand_total_frames = sum(ep_lens)
    print(f"  {len(npz_paths)} episodes, {grand_total_frames} total frames", flush=True)

    for i, npz_path in enumerate(npz_paths):
        d = np.load(npz_path, allow_pickle=True)
        # Materialize arrays once. NpzFile.__getitem__ decompresses lazily, so
        # writing `d["image"][t]` inside the loop re-decompresses the entire
        # array (~25MB) per frame. Pull each array out once.
        images = d["image"]
        wrist_images = d["wrist_image"]
        states = d["state"].astype(np.float32, copy=False)
        actions = d["actions"].astype(np.float32, copy=False)
        T = states.shape[0]
        start_idx = 0
        if args.trim_leading_idle and T > args.idle_min_run + 1:
            xyz_norm = np.linalg.norm(actions[:, :3], axis=1)
            idle = xyz_norm < args.idle_action_thresh
            run = 0
            for t in range(T):
                if idle[t]:
                    run += 1
                else:
                    break
            if run >= args.idle_min_run:
                max_trim = int(T * args.idle_max_trim_frac)
                start_idx = min(run, max_trim)
        images = images[start_idx:]
        wrist_images = wrist_images[start_idx:]
        states = states[start_idx:]
        actions = actions[start_idx:]
        T = states.shape[0]
        # task may be a 0-d numpy array of dtype object/str
        task_arr = d["task"]
        task = task_arr.item() if task_arr.ndim == 0 else str(task_arr)
        if isinstance(task, bytes):
            task = task.decode()

        ep_t0 = time.time()
        for t in range(T):
            dataset.add_frame(
                {
                    "image": images[t],
                    "wrist_image": wrist_images[t],
                    "state": states[t],
                    "actions": actions[t],
                    "task": task,
                }
            )
            total_frames += 1
            if total_frames % 200 == 0:
                elapsed = time.time() - start
                fps_eff = total_frames / max(elapsed, 1e-6)
                eta = (grand_total_frames - total_frames) / max(fps_eff, 1e-6)
                print(
                    f"  ... frame {total_frames}/{grand_total_frames} "
                    f"({100.0*total_frames/grand_total_frames:.1f}%) "
                    f"| {fps_eff:.0f} fps | elapsed {elapsed:.0f}s | eta {eta:.0f}s",
                    flush=True,
                )
        save_t0 = time.time()
        print(f"  [{i+1}/{len(npz_paths)}] {npz_path.name}: T={T} add={save_t0-ep_t0:.1f}s saving...", flush=True)
        dataset.save_episode()
        print(f"  [{i+1}/{len(npz_paths)}] saved in {time.time()-save_t0:.1f}s | task={task!r}", flush=True)

    print(f"Done. {len(npz_paths)} episodes, {total_frames} frames -> {out_path} "
          f"(total {time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
