"""Remove excess episodes so that position_label counts are within 1.15× of the
minimum across left / middle / right.

Each episode is represented by a pair of files in the task directory:
  data_<N>.hdf5
  audio_meta_<N>.json

Usage:
  python scripts/balance_positions.py --task-dir <path-to-task-dir> [--dry-run]
"""

import argparse
import json
import os
import random
from collections import defaultdict


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", required=True,
                   help="Directory containing data_N.hdf5 and audio_meta_N.json files")
    p.add_argument("--max-ratio", type=float, default=1.15,
                   help="Maximum allowed ratio of any direction count to the minimum (default: 1.15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without actually deleting")
    return p.parse_args()


def main():
    args = _parse_args()
    task_dir = args.task_dir

    meta_files = [
        f for f in os.listdir(task_dir)
        if f.startswith("audio_meta_") and f.endswith(".json")
    ]
    if not meta_files:
        print(f"[balance] no audio_meta_*.json found in {task_dir}, nothing to do")
        return

    groups = defaultdict(list)
    for fname in meta_files:
        with open(os.path.join(task_dir, fname)) as fh:
            data = json.load(fh)
        label = data.get("position_label", "unknown")
        ep_id = fname[len("audio_meta_"):-len(".json")]
        groups[label].append(ep_id)

    counts = {lbl: len(ids) for lbl, ids in groups.items()}
    print(f"[balance] before: {counts}")

    min_count = min(counts.values())
    cap = int(min_count * args.max_ratio)

    rng = random.Random(args.seed)
    to_delete = []
    for label, ids in groups.items():
        if len(ids) > cap:
            rng.shuffle(ids)
            to_delete.extend(ids[cap:])

    if not to_delete:
        print(f"[balance] already balanced (min={min_count}, cap={cap}), nothing deleted")
        return

    print(f"[balance] min={min_count}, cap={cap} — deleting {len(to_delete)} episodes")
    for ep_id in sorted(to_delete, key=int):
        json_path = os.path.join(task_dir, f"audio_meta_{ep_id}.json")
        hdf5_path = os.path.join(task_dir, f"data_{ep_id}.hdf5")
        for path in (json_path, hdf5_path):
            if os.path.exists(path):
                if args.dry_run:
                    print(f"  [dry-run] would delete {path}")
                else:
                    os.remove(path)
                    print(f"  deleted {path}")

    if not args.dry_run:
        # Recount to confirm
        remaining = defaultdict(int)
        for fname in os.listdir(task_dir):
            if fname.startswith("audio_meta_") and fname.endswith(".json"):
                with open(os.path.join(task_dir, fname)) as fh:
                    label = json.load(fh).get("position_label", "unknown")
                remaining[label] += 1
        print(f"[balance] after:  {dict(remaining)}")


if __name__ == "__main__":
    main()
