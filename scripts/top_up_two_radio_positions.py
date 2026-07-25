"""Top up select_radio_two episodes to balanced target-position counts.

The generator can fail an episode without saving HDF5. This wrapper assigns
each worker a fixed number of successful episodes to produce, retries failed
attempts with fresh IDs, and removes failed demo videos for those attempts.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


LABELS = ("left", "middle", "right")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default="/home/rllab/Desktop/AVLABench")
    p.add_argument("--dataset-dir", default="dataset_oracle_two")
    p.add_argument("--task-name", default="select_radio_two")
    p.add_argument("--target-count", type=int, default=None)
    p.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    p.add_argument("--workers-left", type=int, default=4)
    p.add_argument("--workers-middle", type=int, default=1)
    p.add_argument("--workers-right", type=int, default=2)
    p.add_argument("--start-id", type=int, default=None)
    p.add_argument("--start-idle-seconds", type=float, default=2.0)
    p.add_argument("--dataset-fps", type=int, default=10)
    p.add_argument("--robot", default="franka")
    p.add_argument("--max-attempts-per-success", type=int, default=50)
    return p.parse_args()


def _task_dir(args):
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(args.repo_root, dataset_dir)
    return os.path.join(dataset_dir, args.task_name)


def _episode_id(fname):
    m = re.match(r"(?:data|audio_meta|demo)_(\d+)", fname)
    return int(m.group(1)) if m else None


def _count_successes(task_dir):
    counts = Counter()
    ids_by_label = defaultdict(list)
    if not os.path.exists(task_dir):
        return counts, ids_by_label
    for fname in os.listdir(task_dir):
        if not fname.startswith("audio_meta_") or not fname.endswith(".json"):
            continue
        ep_id = fname[len("audio_meta_") : -len(".json")]
        if not os.path.exists(os.path.join(task_dir, f"data_{ep_id}.hdf5")):
            continue
        with open(os.path.join(task_dir, fname)) as fh:
            label = json.load(fh).get("position_label", "unknown")
        counts[label] += 1
        ids_by_label[label].append(int(ep_id))
    return counts, ids_by_label


def _max_existing_id(task_dir):
    max_id = -1
    if not os.path.exists(task_dir):
        return max_id
    for fname in os.listdir(task_dir):
        ep_id = _episode_id(fname)
        if ep_id is not None:
            max_id = max(max_id, ep_id)
    return max_id


def _cleanup_failed_outputs(task_dir, ep_id):
    for suffix in (
        f"demo_{ep_id}_success_False.mp4",
        f"data_{ep_id}.hdf5",
        f"audio_meta_{ep_id}.json",
    ):
        path = os.path.join(task_dir, suffix)
        if os.path.exists(path):
            os.remove(path)


def _success_for(task_dir, ep_id, label):
    h5 = os.path.join(task_dir, f"data_{ep_id}.hdf5")
    meta = os.path.join(task_dir, f"audio_meta_{ep_id}.json")
    if not os.path.exists(h5) or not os.path.exists(meta):
        return False
    with open(meta) as fh:
        return json.load(fh).get("position_label") == label


def _run_worker(payload):
    args_dict, label, worker_idx, quota, first_id, stride = payload
    repo_root = args_dict["repo_root"]
    task_dir = args_dict["task_dir"]
    dataset_dir = args_dict["dataset_dir"]
    max_attempts = args_dict["max_attempts_per_success"]
    made = 0
    attempts = 0
    ep_id = first_id + worker_idx
    logs = []

    while made < quota:
        if attempts >= quota * max_attempts:
            raise RuntimeError(
                f"{label} worker {worker_idx}: made {made}/{quota} after "
                f"{attempts} attempts"
            )
        attempts += 1
        cmd = [
            sys.executable,
            "scripts/trajectory_generation.py",
            "--task-name",
            args_dict["task_name"],
            "--oracle-mode",
            "--save-dir",
            dataset_dir,
            "--n-sample",
            "1",
            "--start-id",
            str(ep_id),
            "--max-episode",
            "1000000000",
            "--start-idle-seconds",
            str(args_dict["start_idle_seconds"]),
            "--dataset-fps",
            str(args_dict["dataset_fps"]),
            "--robot",
            args_dict["robot"],
            "--target-position-label",
            label,
        ]
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ok = proc.returncode == 0 and _success_for(task_dir, ep_id, label)
        if ok:
            made += 1
            logs.append(
                f"{label} worker {worker_idx}: success {made}/{quota} "
                f"id={ep_id} attempts={attempts} elapsed={time.time() - t0:.1f}s"
            )
        else:
            _cleanup_failed_outputs(task_dir, ep_id)
            tail = "\n".join(proc.stdout.splitlines()[-8:])
            logs.append(
                f"{label} worker {worker_idx}: discard id={ep_id} "
                f"rc={proc.returncode}\n{tail}"
            )
        ep_id += stride
    return {"label": label, "worker": worker_idx, "made": made, "attempts": attempts, "logs": logs}


def _split_quota(total, workers):
    workers = max(1, min(workers, total))
    base, rem = divmod(total, workers)
    return [base + (1 if i < rem else 0) for i in range(workers)]


def main():
    args = _parse_args()
    os.chdir(args.repo_root)
    task_dir = _task_dir(args)
    counts, _ = _count_successes(task_dir)
    target = args.target_count
    if target is None:
        target = max(counts[label] for label in LABELS)
    needed = {label: max(0, target - counts[label]) for label in args.labels}
    print(f"[top-up] before={dict(counts)} target={target} needed={needed}", flush=True)

    start_id = args.start_id
    if start_id is None:
        start_id = _max_existing_id(task_dir) + 1
    n_workers = {
        "left": args.workers_left,
        "middle": args.workers_middle,
        "right": args.workers_right,
    }
    stride = 1000
    jobs = []
    label_offsets = {"left": 0, "middle": 100000, "right": 200000}
    args_dict = vars(args).copy()
    args_dict["task_dir"] = task_dir
    if not os.path.isabs(args_dict["dataset_dir"]):
        args_dict["dataset_dir"] = os.path.join(args.repo_root, args.dataset_dir)

    for label, total in needed.items():
        if total <= 0:
            continue
        # Use one-success jobs so progress is reported after every successful
        # episode while ProcessPoolExecutor limits actual concurrency.
        quotas = [1] * total
        first_id = start_id + label_offsets[label]
        for worker_idx, quota in enumerate(quotas):
            jobs.append((args_dict, label, worker_idx, quota, first_id, stride))

    if not jobs:
        print("[top-up] already balanced; nothing to do")
        return

    max_parallel = sum(n_workers[label] for label, total in needed.items() if total > 0)
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_run_worker, job) for job in jobs]
        for fut in as_completed(futures):
            result = fut.result()
            print(
                f"[top-up] done {result['label']} worker {result['worker']}: "
                f"made={result['made']} attempts={result['attempts']}",
                flush=True,
            )
            for line in result["logs"][-5:]:
                print(f"  {line}", flush=True)

    after, _ = _count_successes(task_dir)
    print(f"[top-up] after={dict(after)}", flush=True)
    short = {label: target - after[label] for label in LABELS if after[label] < target}
    if short:
        raise SystemExit(f"[top-up] still short: {short}")


if __name__ == "__main__":
    main()
