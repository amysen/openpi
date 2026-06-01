"""Merge multiple partial LeRobot shard datasets into one.

Each shard was produced by convert_composuite_npz_to_lerobot.py with
--start-idx / --end-idx. Images are stored as bytes inside the parquet
files, so no external image directories need to be moved.

Usage (inside openpi .venv):
  python experiments/merge_lerobot_shards.py \
      --shard-prefix composuite_plate_osc_n300_trimmed_shard \
      --num-shards 12 \
      --output-id composuite_plate_osc_n300_trimmed \
      [--overwrite] [--delete-shards]
"""

import argparse
import json
import pathlib
import shutil

import pandas as pd
from datasets import Dataset, Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

CHUNKS_SIZE = 1000

# Columns that LeRobot expects to be typed as datasets.Image() so they get
# PIL-decoded on access. A bare df.to_parquet() preserves the struct<bytes,path>
# data but drops the `huggingface` schema metadata, leaving downstream loaders
# returning raw dicts that torch.tensor() chokes on. We round-trip through
# datasets.Dataset to restore the feature.
IMAGE_COLUMNS = ("image", "wrist_image")


def chunk_dir(base: pathlib.Path, episode_index: int) -> pathlib.Path:
    chunk = episode_index // CHUNKS_SIZE
    return base / "data" / f"chunk-{chunk:03d}"


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard-ids", nargs="+",
                       help="Explicit list of shard repo-ids in merge order.")
    group.add_argument("--shard-prefix", metavar="PREFIX",
                       help="Shard repo-ids are PREFIX_0, PREFIX_1, ... PREFIX_(N-1).")
    p.add_argument("--num-shards", type=int,
                   help="Required when using --shard-prefix.")
    p.add_argument("--output-id", required=True,
                   help="Repo-id for the merged output dataset.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--delete-shards", action="store_true",
                   help="Remove shard datasets after a successful merge.")
    args = p.parse_args()

    if args.shard_prefix is not None:
        if args.num_shards is None:
            p.error("--num-shards is required with --shard-prefix")
        shard_ids = [f"{args.shard_prefix}_{i}" for i in range(args.num_shards)]
    else:
        shard_ids = args.shard_ids

    out_path = HF_LEROBOT_HOME / args.output_id
    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {out_path}. Use --overwrite.")
        shutil.rmtree(out_path)
    (out_path / "meta").mkdir(parents=True)

    episode_offset = 0
    frame_offset = 0
    all_episodes_meta = []
    all_episodes_stats = []
    first_info = None

    for shard_id in shard_ids:
        shard_path = HF_LEROBOT_HOME / shard_id
        if not shard_path.exists():
            raise SystemExit(f"Shard not found: {shard_path}")

        info = json.loads((shard_path / "meta" / "info.json").read_text())
        if first_info is None:
            first_info = info

        parquets = sorted((shard_path / "data").rglob("*.parquet"))
        print(f"  {shard_id}: {len(parquets)} episodes, {info['total_frames']} frames")

        for pq_path in parquets:
            df = pd.read_parquet(pq_path)
            df["episode_index"] = df["episode_index"] + episode_offset
            df["index"] = df["index"] + frame_offset
            # frame_index is per-episode, leave it unchanged

            new_ep_idx = int(df["episode_index"].iloc[0])
            out_dir = chunk_dir(out_path, new_ep_idx)
            out_dir.mkdir(parents=True, exist_ok=True)
            ds = Dataset.from_pandas(df, preserve_index=False)
            for col in IMAGE_COLUMNS:
                if col in ds.column_names:
                    ds = ds.cast_column(col, Image())
            ds.to_parquet(out_dir / f"episode_{new_ep_idx:06d}.parquet")

        episodes_meta = [
            json.loads(line)
            for line in (shard_path / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for m in episodes_meta:
            m["episode_index"] += episode_offset
            all_episodes_meta.append(m)

        episodes_stats = [
            json.loads(line)
            for line in (shard_path / "meta" / "episodes_stats.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for s in episodes_stats:
            s["episode_index"] += episode_offset
            all_episodes_stats.append(s)

        episode_offset += info["total_episodes"]
        frame_offset += info["total_frames"]

    total_episodes = episode_offset
    total_frames = frame_offset

    with open(out_path / "meta" / "episodes.jsonl", "w") as f:
        for m in all_episodes_meta:
            f.write(json.dumps(m) + "\n")

    with open(out_path / "meta" / "episodes_stats.jsonl", "w") as f:
        for s in all_episodes_stats:
            f.write(json.dumps(s) + "\n")

    first_shard_path = HF_LEROBOT_HOME / shard_ids[0]
    shutil.copy2(first_shard_path / "meta" / "tasks.jsonl", out_path / "meta" / "tasks.jsonl")

    merged_info = dict(first_info)
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_chunks"] = (total_episodes - 1) // CHUNKS_SIZE + 1
    merged_info["splits"] = {"train": f"0:{total_episodes}"}
    (out_path / "meta" / "info.json").write_text(json.dumps(merged_info, indent=4))

    print(f"Merged {len(shard_ids)} shards: {total_episodes} episodes, {total_frames} frames -> {out_path}")

    if args.delete_shards:
        for shard_id in shard_ids:
            shutil.rmtree(HF_LEROBOT_HOME / shard_id)
            print(f"  Deleted shard: {shard_id}")


if __name__ == "__main__":
    main()
