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


def _leading_idle_trim(actions, thresh, min_run, max_trim_frac):
    """Return the start index after the leading near-zero-action run."""
    T = actions.shape[0]
    if T <= min_run + 1:
        return 0
    xyz_norm = np.linalg.norm(actions[:, :3], axis=1)
    run = 0
    for t in range(T):
        if xyz_norm[t] < thresh:
            run += 1
        else:
            break
    if run >= min_run:
        return min(run, int(T * max_trim_frac))
    return 0


def _retime_episode(actions, target_pos_norm, target_rot_norm, max_merge):
    """Merge runs of consecutive low-motion frames into single frames whose
    action is the accumulated commanded delta over the run.

    The scripted collector's P-controller emits long stretches of tiny deltas
    as it converges on each waypoint (plus 45-step hold-closes); BC on that
    distribution teaches the policy to under-act. Greedily accumulate
    consecutive actions until the summed |Δpos| (or |Δrot|) reaches the target
    magnitude, a gripper-command change is hit, or max_merge frames are
    merged — then emit one frame. Observations come from the first frame of
    each span, so (obs_t, sum of commands issued from obs_t) stays a valid
    (state, action) pair. Fast frames (already ≥ target) pass through 1:1;
    slow segments densify. Gripper transitions are never merged across, and
    max_merge keeps a few frames inside hold phases so the policy still
    learns to wait for the pinch to seat.

    Returns (keep_indices, merged_actions).
    """
    T = actions.shape[0]
    keep, merged = [], []
    i = 0
    while i < T:
        acc = actions[i].copy()
        j = i + 1
        while (
            j < T
            and (j - i) < max_merge
            and abs(actions[j, -1] - actions[i, -1]) < 0.5
            and np.linalg.norm(acc[:3]) < target_pos_norm
            and np.linalg.norm(acc[3:6]) < target_rot_norm
        ):
            acc[:6] += actions[j, :6]
            j += 1
        acc[:6] = np.clip(acc[:6], -1.0, 1.0)
        keep.append(i)
        merged.append(acc)
        i = j
    return np.asarray(keep, dtype=np.int64), np.stack(merged).astype(np.float32)


# Sub-task language annotations for the plate-transfer pipeline. Pi-0.5 was
# pretrained with high-level subtask decomposition; one monolithic prompt over
# ~800 steps gives it nothing to plan with. Phase names come from the
# collector's `phase` NPZ array; each segment becomes its own LeRobot episode
# with a shorter horizon and its own instruction.
PLATE_SUBTASK_SEGMENTS = [
    ("pull the plate off the stand",
     {"orient_wrist", "rotate_above", "descend_outside", "slide_in",
      "rise_to_grasp", "close", "slide_off"}),
    ("carry the plate over to the second stand",
     {"yaw_for_carry", "lift_carry", "traverse_y", "align_x"}),
    ("set the plate down on top of the second stand",
     {"settle_over_goal", "stabilize_over_shelf2", "descend_shelf2",
      "open", "retract_clear"}),
]


def _subtask_slices(actions, states, phases):
    """Return [(task_string_or_None, slice)] sub-segments of one episode.

    With per-frame phase labels (collector NPZs written after 2026-07): map
    phases onto PLATE_SUBTASK_SEGMENTS. Without them: fall back to a 2-way
    kinematic split at the pull-off point (first frame after the gripper
    closes where the eef has retreated >15 cm in -x, i.e. slide_off done).
    A task of None means "keep the episode's original prompt".
    """
    T = actions.shape[0]
    if phases is not None and len(phases) == T:
        seg_of = {}
        for i, (_task, names) in enumerate(PLATE_SUBTASK_SEGMENTS):
            for n in names:
                seg_of[n] = i
        ids = [seg_of.get(str(ph), -1) for ph in phases]
        out, s = [], 0
        for t in range(1, T + 1):
            if t == T or (ids[t] != ids[s] and ids[t] != -1 and ids[s] != -1):
                task = PLATE_SUBTASK_SEGMENTS[ids[s]][0] if ids[s] != -1 else None
                out.append((task, slice(s, t)))
                s = t
        # Merge unmapped runs (e.g. "preview") into the previous segment.
        merged = []
        for task, sl in out:
            if task is None and merged:
                prev_task, prev_sl = merged[-1]
                merged[-1] = (prev_task, slice(prev_sl.start, sl.stop))
            else:
                merged.append((task, sl))
        return merged
    # Fallback: gripper-close + eef-x retreat heuristic (2 segments).
    grip = actions[:, -1]
    closed = np.where(grip > 0)[0]
    if closed.size == 0:
        return [(None, slice(0, T))]
    c0 = int(closed[0])
    x_at_close = states[c0, 0]
    pulled = np.where(states[c0:, 0] < x_at_close - 0.15)[0]
    if pulled.size == 0:
        return [(None, slice(0, T))]
    split = c0 + int(pulled[0])
    return [
        (PLATE_SUBTASK_SEGMENTS[0][0], slice(0, split)),
        ("place the plate on top of the second stand", slice(split, T)),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-dir", required=True,
                   help="Directory containing *.npz episode files")
    p.add_argument("--repo-id", default="composuite_box_objective_v1")
    p.add_argument("--robot-type", default="panda")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--overwrite", action="store_true",
                   help="Delete the output directory if it already exists")
    # Threads-only staging (processes=0): multi-PROCESS writers race
    # save_episode's rmtree of the images staging dir on NFS (deleted-but-open
    # files linger as .nfsXXXX silly-renames -> "Directory not empty"). With
    # threads, every file handle lives in this process and wait_until_done()
    # is airtight.
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--image-writer-processes", type=int, default=0)
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
    p.add_argument(
        "--retime",
        action="store_true",
        help="Merge consecutive low-motion frames so each retained frame carries a "
             "larger accumulated action (attacks under-acting learned from the "
             "P-controller's convergence tails). See _retime_episode.",
    )
    p.add_argument(
        "--retime-target-pos-norm", type=float, default=0.5,
        help="Accumulate consecutive actions until |sum Δpos| (normalized OSC units, "
             "1.0 = controller output_max) reaches this, then emit a frame.",
    )
    p.add_argument(
        "--retime-target-rot-norm", type=float, default=0.5,
        help="Also emit once accumulated |Δrot| reaches this (axis-angle sum approx).",
    )
    p.add_argument(
        "--retime-max-merge", type=int, default=6,
        help="Max input frames merged per output frame. Bounds timescale distortion "
             "and keeps several frames inside gripper hold phases.",
    )
    p.add_argument(
        "--retime-dry-run",
        action="store_true",
        help="Print per-episode retiming stats (frames in/out, action magnitudes) and "
             "exit without writing any dataset. Use to tune the retime targets.",
    )
    p.add_argument(
        "--subtask-split", default="off", choices=["off", "segments", "mixed"],
        help="Split each episode into sub-task segments with their own language "
             "prompts (phase labels when present, kinematic fallback otherwise). "
             "'segments' emits only the segments; 'mixed' also keeps the full "
             "episode under its original prompt (recommended: full-prompt eval "
             "still matches the training distribution).",
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

    if args.retime_dry_run:
        tot_in = tot_out = 0
        pre_norms, post_norms = [], []
        for pth in npz_paths:
            with np.load(pth, allow_pickle=True) as d:
                actions = d["actions"].astype(np.float32)
            start_idx = 0
            if args.trim_leading_idle:
                start_idx = _leading_idle_trim(
                    actions, args.idle_action_thresh, args.idle_min_run, args.idle_max_trim_frac)
            actions = actions[start_idx:]
            keep, merged = _retime_episode(
                actions, args.retime_target_pos_norm, args.retime_target_rot_norm,
                args.retime_max_merge)
            tot_in += actions.shape[0]
            tot_out += merged.shape[0]
            pre_norms.append(np.linalg.norm(actions[:, :3], axis=1))
            post_norms.append(np.linalg.norm(merged[:, :3], axis=1))
            print(f"  {pth.name}: trim={start_idx} frames {actions.shape[0]} -> {merged.shape[0]} "
                  f"({actions.shape[0]/max(merged.shape[0],1):.2f}x)")
        pre = np.concatenate(pre_norms); post = np.concatenate(post_norms)
        print(f"\nTOTAL frames: {tot_in} -> {tot_out} ({tot_in/max(tot_out,1):.2f}x compression)")
        for name, a in (("before", pre), ("after", post)):
            nz = a[a > 1e-6]
            print(f"|dpos| {name}: mean={a.mean():.3f} median={np.median(a):.3f} "
                  f"p90={np.percentile(a, 90):.3f} zero-frac={(a <= 1e-6).mean():.3f} "
                  f"nonzero-median={np.median(nz) if nz.size else 0:.3f}")
        return

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
        phases = d["phase"] if "phase" in d.files else None
        T = states.shape[0]
        start_idx = 0
        if args.trim_leading_idle:
            start_idx = _leading_idle_trim(
                actions, args.idle_action_thresh, args.idle_min_run, args.idle_max_trim_frac)
        images = images[start_idx:]
        wrist_images = wrist_images[start_idx:]
        states = states[start_idx:]
        actions = actions[start_idx:]
        if phases is not None:
            phases = phases[start_idx:]
        # task may be a 0-d numpy array of dtype object/str
        task_arr = d["task"]
        task = task_arr.item() if task_arr.ndim == 0 else str(task_arr)
        if isinstance(task, bytes):
            task = task.decode()

        # Decide what to emit: the full episode, sub-task segments, or both.
        # Splits happen on raw frames (phase labels align to them); retiming is
        # applied per emitted episode so no merge crosses a segment boundary.
        emissions = []
        if args.subtask_split in ("segments", "mixed"):
            for seg_task, sl in _subtask_slices(actions, states, phases):
                if sl.stop - sl.start >= 10:
                    emissions.append((seg_task or task, sl))
        if args.subtask_split in ("off", "mixed") or not emissions:
            emissions.insert(0, (task, slice(0, states.shape[0])))

        ep_t0 = time.time()
        n_emitted = 0
        for ep_task, sl in emissions:
            e_images, e_wrists = images[sl], wrist_images[sl]
            e_states, e_actions = states[sl], actions[sl]
            if args.retime:
                keep, e_actions = _retime_episode(
                    e_actions, args.retime_target_pos_norm, args.retime_target_rot_norm,
                    args.retime_max_merge)
                e_images, e_wrists, e_states = e_images[keep], e_wrists[keep], e_states[keep]
            for t in range(e_states.shape[0]):
                dataset.add_frame(
                    {
                        "image": e_images[t],
                        "wrist_image": e_wrists[t],
                        "state": e_states[t],
                        "actions": e_actions[t],
                        "task": ep_task,
                    }
                )
                total_frames += 1
                if total_frames % 200 == 0:
                    elapsed = time.time() - start
                    fps_eff = total_frames / max(elapsed, 1e-6)
                    eta = (grand_total_frames - total_frames) / max(fps_eff, 1e-6)
                    print(
                        f"  ... frame {total_frames}/~{grand_total_frames} "
                        f"| {fps_eff:.0f} fps | elapsed {elapsed:.0f}s | eta {eta:.0f}s",
                        flush=True,
                    )
            # save_episode() rmtree's the images staging dir WITHOUT waiting for
            # the async image writer (lerobot_dataset.py ~L905). With long
            # episodes the writer drains naturally, but short sub-task segments
            # save every ~100 frames and race it (OSError: Directory not empty).
            # Drain explicitly before every save.
            if dataset.image_writer is not None:
                dataset.image_writer.wait_until_done()
            try:
                dataset.save_episode()
            except OSError as e:
                # NFS can still fail the staging-dir rmtree transiently
                # (attribute-cache lag / .nfs silly-renames). At that point the
                # episode is already fully persisted (parquet + meta written
                # before the rmtree; only the buffer reset comes after), so
                # recover: settle, sweep the staging dir, reset the buffer.
                if "Directory not empty" not in str(e):
                    raise
                time.sleep(1.0)
                shutil.rmtree(dataset.root / "images", ignore_errors=True)
                dataset.episode_buffer = dataset.create_episode_buffer()
            n_emitted += 1
        print(f"  [{i+1}/{len(npz_paths)}] {npz_path.name}: emitted {n_emitted} episode(s) "
              f"in {time.time()-ep_t0:.1f}s "
              f"({'phase-labeled' if phases is not None else 'heuristic/full'})", flush=True)

    print(f"Done. {len(npz_paths)} episodes, {total_frames} frames -> {out_path} "
          f"(total {time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
