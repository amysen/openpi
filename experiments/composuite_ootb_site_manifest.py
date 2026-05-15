"""Build the CompoSuite OOTB baseline website manifest from smoke JSONL."""

import argparse
import collections
import json
import pathlib
import shutil
import statistics
from datetime import date


SLICE_LABELS = {
    "known-ish": "Known-ish Skills",
    "new-clutter": "New Clutter",
    "new-object": "New Object",
    "new-objective": "New Objective",
    "new-obstacle": "New Obstacle",
    "new-robot": "New Robot",
    "mixed": "Mixed Composition",
}


def classify(row):
    robot = row["robot"]
    obj = row["object"]
    obstacle = row["obstacle"]
    objective = row["objective"]
    clutter = row.get("clutter", "none")
    if robot != "Panda":
        return "new-robot"
    if clutter != "none":
        return "new-clutter"
    if obstacle != "none":
        return "new-obstacle"
    if obj != "box":
        return "new-object"
    if objective not in {"pick_and_place", "push"}:
        return "new-objective"
    return "known-ish"


def looks_like_predicate_artifact(group):
    if not group:
        return False
    if not all(row.get("success") for row in group):
        return False
    mean_steps = statistics.mean(float(row.get("steps", 0)) for row in group)
    mean_return = statistics.mean(float(row.get("return", 0)) for row in group)
    return mean_steps <= 10.5 and abs(mean_return) < 1e-6


def invalid_objective(objective):
    return objective == "push"


def needs_manual_audit(obj):
    return False


def failure_mode(successes, trials, group, objective, obj):
    if invalid_objective(objective):
        return "Invalid for OOTB claims: videos show pick/place-style behavior, not planar pushing; push predicate needs audit."
    if looks_like_predicate_artifact(group):
        return "Success predicate fires immediately with zero return; mark as invalid until audited."
    if trials <= 0:
        return "No trials recorded."
    rate = successes / trials
    if rate >= 0.8 and trials >= 20:
        return "Reliable success across seeds."
    if successes > 0:
        return "Occasional success; not robust at this sample size."
    return "No task successes; inspect video for approach/contact quality."


def copy_video(video, out_path, robot, obj, obstacle, objective, clutter, label):
    if not video:
        return ""
    src = pathlib.Path(video)
    if not src.exists():
        return video
    safe_name = f"{robot}_{obj}_{obstacle}_{objective}_{clutter}_{label}_{src.name}"
    dst = pathlib.Path(out_path).parent / "videos" / safe_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"videos/{safe_name}"


def representative_videos(group, out_path, robot, obj, obstacle, objective, clutter, copy_videos):
    videos = []
    success_row = next((row for row in group if row.get("success") and row.get("video")), None)
    failure_row = next((row for row in group if not row.get("success") and row.get("video")), None)

    for label, row in (("success", success_row), ("failure", failure_row)):
        if row is None:
            continue
        display_label = "invalid" if objective == "push" and label == "success" else label
        if obj == "dumbbell" and label == "success":
            display_label = "audit"
        video = row.get("video", "")
        if copy_videos:
            video = copy_video(video, out_path, robot, obj, obstacle, objective, clutter, display_label)
        videos.append(
            {
                "label": display_label,
                "path": video,
                "episode": row.get("episode"),
                "steps": row.get("steps"),
                "return": row.get("return"),
                "diag": row.get("diag", {}),
            }
        )

    if not videos:
        row = group[0] if group else {}
        video = row.get("video", "")
        if copy_videos:
            video = copy_video(video, out_path, robot, obj, obstacle, objective, clutter, "representative")
        if video:
            videos.append(
                {
                    "label": "representative",
                    "path": video,
                    "episode": row.get("episode"),
                    "steps": row.get("steps"),
                    "return": row.get("return"),
                    "diag": row.get("diag", {}),
                }
            )

    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        required=True,
        nargs="+",
        help="One or more composuite_smoke.py results JSONL files",
    )
    parser.add_argument("--model-id", default="pi05-libero")
    parser.add_argument("--model-name", default="Pi-0.5 LIBERO")
    parser.add_argument("--checkpoint", default="physical-intelligence pi05_libero")
    parser.add_argument("--out", default="docs/composuite_ootb_baseline/results.json")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy representative videos next to the website and rewrite manifest paths.",
    )
    args = parser.parse_args()

    rows = []
    for jsonl_path in args.jsonl:
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if invalid_objective(row.get("objective")):
                        continue
                    rows.append(row)

    grouped = collections.defaultdict(list)
    for row in rows:
        key = (
            row["robot"],
            row["object"],
            row["obstacle"],
            row["objective"],
            row.get("clutter", "none"),
            row.get("prompt", row.get("generated_language", "")),
        )
        grouped[key].append(row)

    results = []
    used_slices = set()
    for (robot, obj, obstacle, objective, clutter, prompt), group in sorted(grouped.items()):
        env_successes = sum(1 for row in group if row.get("success"))
        trials = len(group)
        successes = 0 if needs_manual_audit(obj) else env_successes
        slice_id = classify(group[0])
        used_slices.add(slice_id)
        representative = next((row for row in group if row.get("success")), group[0])

        videos = representative_videos(
            group, args.out, robot, obj, obstacle, objective, clutter, args.copy_videos
        )
        video = videos[0]["path"] if videos else representative.get("video", "")

        results.append(
            {
                "model_id": args.model_id,
                "slice": slice_id,
                "robot": robot,
                "object": obj,
                "obstacle": obstacle,
                "objective": objective,
                "clutter": clutter,
                "prompt": prompt,
                "generated_language": representative.get("generated_language", prompt),
                "trials": trials,
                "successes": successes,
                "env_successes": env_successes,
                "failure_mode": failure_mode(successes, trials, group, objective, obj),
                "video": video,
                "videos": videos,
            }
        )

    manifest = {
        "benchmark": "LIBERO-CompoSuite OOTB VLA Baseline",
        "updated": date.today().isoformat(),
        "protocol": {
            "checkpoint": "pi05_libero base checkpoint",
            "training": "none; frozen out-of-the-box evaluation",
            "controller": "OSC_POSE",
            "cameras": "agentview + wrist, 256x256",
            "state": "8-D [eef_pos, axis_angle, gripper_qpos[:2]]",
            "action": "7-D [delta_pos, delta_axis_angle, gripper]",
            "primary_metric": "environment success / done",
            "recommended_trials_for_claims": "20-50 per task",
        },
        "slices": [
            {
                "id": slice_id,
                "label": SLICE_LABELS[slice_id],
                "description": "",
            }
            for slice_id in sorted(used_slices)
        ],
        "models": [
            {
                "id": args.model_id,
                "name": args.model_name,
                "checkpoint": args.checkpoint,
                "notes": "Frozen checkpoint; no fine-tuning on CompoSuite demos.",
            }
        ],
        "results": results,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(results)} task rows from {len(rows)} rollouts to {out}")


if __name__ == "__main__":
    main()
