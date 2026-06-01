"""Run frozen-checkpoint VLA sweeps on LIBERO-CompoSuite.

This is a thin orchestrator around experiments/composuite_smoke.py. It keeps
task presets and output paths consistent so results can be pushed directly into
docs/composuite_ootb_baseline/results.json.
"""

import argparse
import pathlib
import subprocess
import sys


ROBOTS = ["Panda", "IIWA", "Jaco", "Gen3"]
OBJECTS = ["box", "hollow_box", "plate", "dumbbell"]
OBSTACLES = ["none", "object_wall", "object_door", "goal_wall"]
OBJECTIVES = ["pick_and_place", "push", "trash_can", "shelf"]
VALID_OBJECTIVES = ["pick_and_place", "trash_can", "shelf"]
CLUTTER = ["none", "all_objects", "decoys", "random"]


PRESETS = {
    # Fastest real baseline: object/objective composition with the Panda and no obstacles.
    "core16": {
        "robots": ["Panda"],
        "objects": OBJECTS,
        "obstacles": ["none"],
        "objectives": OBJECTIVES,
        "clutters": ["none"],
    },
    # Public-report core sweep with invalid push objective removed.
    "core12": {
        "robots": ["Panda"],
        "objects": OBJECTS,
        "obstacles": ["none"],
        "objectives": VALID_OBJECTIVES,
        "clutters": ["none"],
    },
    # Obstacle axis sweep: same valid objectives, all obstacle variants.
    # This includes "none" as the baseline row beside the three obstacle conditions.
    "obstacles48": {
        "robots": ["Panda"],
        "objects": OBJECTS,
        "obstacles": OBSTACLES,
        "objectives": VALID_OBJECTIVES,
        "clutters": ["none"],
    },
    # Smaller obstacle sweep for quick iteration before obstacles48.
    "box_obstacles12": {
        "robots": ["Panda"],
        "objects": ["box"],
        "obstacles": OBSTACLES,
        "objectives": VALID_OBJECTIVES,
        "clutters": ["none"],
    },
    # Axis slices that are good for a first PI-facing report.
    "report36": {
        "robots": ["Panda", "IIWA", "Jaco", "Gen3"],
        "objects": ["box", "plate", "dumbbell"],
        "obstacles": ["none"],
        "objectives": ["pick_and_place", "push", "trash_can"],
        "clutters": ["none"],
    },
    # All Panda tasks across object, obstacle, objective.
    "panda64": {
        "robots": ["Panda"],
        "objects": OBJECTS,
        "obstacles": OBSTACLES,
        "objectives": OBJECTIVES,
        "clutters": ["none"],
    },
    "panda48": {
        "robots": ["Panda"],
        "objects": OBJECTS,
        "obstacles": OBSTACLES,
        "objectives": VALID_OBJECTIVES,
        "clutters": ["none"],
    },
    # Clutter axis sweep: Panda + 3 valid objectives + no obstacles + 4 clutter modes.
    # 4 objects x 3 objectives x 4 clutter = 48 tasks.
    "clutter48": {
        "robots": ["Panda"],
        "objects": OBJECTS,
        "obstacles": ["none"],
        "objectives": VALID_OBJECTIVES,
        "clutters": CLUTTER,
    },
    # Smaller clutter sweep for quick iteration: just the box.
    "box_clutter12": {
        "robots": ["Panda"],
        "objects": ["box"],
        "obstacles": ["none"],
        "objectives": VALID_OBJECTIVES,
        "clutters": CLUTTER,
    },
    # Plate-focused clutter sweep: pairs nicely with the `decoys` mode.
    "plate_clutter12": {
        "robots": ["Panda"],
        "objects": ["plate"],
        "obstacles": ["none"],
        "objectives": VALID_OBJECTIVES,
        "clutters": CLUTTER,
    },
    # Entire generated benchmark. This is expensive.
    "full256": {
        "robots": ROBOTS,
        "objects": OBJECTS,
        "obstacles": OBSTACLES,
        "objectives": OBJECTIVES,
        "clutters": ["none"],
    },
    # Full benchmark including clutter axis: 4*4*4*4*4 = 1024 tasks.
    "full1024": {
        "robots": ROBOTS,
        "objects": OBJECTS,
        "obstacles": OBSTACLES,
        "objectives": OBJECTIVES,
        "clutters": CLUTTER,
    },
}


def task_specs(preset_name):
    preset = PRESETS[preset_name]
    clutters = preset.get("clutters", ["none"])
    specs = []
    for robot in preset["robots"]:
        for obj in preset["objects"]:
            for obstacle in preset["obstacles"]:
                for objective in preset["objectives"]:
                    for clutter in clutters:
                        if clutter == "none":
                            # Keep legacy 4-field spec when clutter axis is
                            # not exercised so existing run names match.
                            specs.append(f"{robot},{obj},{obstacle},{objective}")
                        else:
                            specs.append(
                                f"{robot},{obj},{obstacle},{objective},{clutter}"
                            )
    return specs


def read_task_specs(path):
    specs = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            specs.append(line)
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="core16")
    parser.add_argument("--model-id", default="pi05-libero")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--out-root", default="data/composuite/ootb")
    parser.add_argument(
        "--task-specs-file",
        default=None,
        help="Optional newline-delimited task spec file for running a subset of the preset.",
    )
    parser.add_argument(
        "--results-jsonl",
        default=None,
        help="Optional results JSONL path. Defaults to results.jsonl under the run directory.",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional worker label used to avoid command/task file collisions.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for composuite_smoke.py. Use the libero conda env when running the simulator client.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_name = f"{args.model_id}_{args.preset}_n{args.trials}_seed{args.seed}"
    out_root = pathlib.Path(args.out_root) / run_name
    video_dir = out_root / "videos"
    if args.results_jsonl:
        results_jsonl = pathlib.Path(args.results_jsonl)
    elif args.worker_id is not None:
        results_jsonl = out_root / f"results_worker_{args.worker_id}.jsonl"
    else:
        results_jsonl = out_root / "results.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)

    specs = read_task_specs(args.task_specs_file) if args.task_specs_file else task_specs(args.preset)
    if not specs:
        raise SystemExit("No task specs to run")
    print(f"Preset: {args.preset}")
    print(f"Tasks: {len(specs)}")
    print(f"Trials per task: {args.trials}")
    print(f"Total rollouts: {len(specs) * args.trials}")
    print(f"Output: {out_root}")

    cmd = [
        args.python,
        "experiments/composuite_smoke.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--task-specs",
        *specs,
        "--num-trials-per-task",
        str(args.trials),
        "--max-steps",
        str(args.max_steps),
        "--replan-steps",
        str(args.replan_steps),
        "--video-out-path",
        str(video_dir),
        "--results-jsonl",
        str(results_jsonl),
        "--seed",
        str(args.seed),
    ]

    file_suffix = f"_worker_{args.worker_id}" if args.worker_id is not None else ""
    (out_root / f"command{file_suffix}.txt").write_text(" ".join(cmd) + "\n")
    (out_root / f"task_specs{file_suffix}.txt").write_text("\n".join(specs) + "\n")

    if args.dry_run:
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
