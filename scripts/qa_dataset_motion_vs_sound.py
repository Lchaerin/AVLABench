#!/usr/bin/env python3
"""Dataset QA: does the expert WAIT for the chime before moving?

Reads a converted LeRobot dataset and, per episode, finds the frame the audio
label first becomes active (chime onset) and the frame the robot state first
deviates from rest (motion onset), then reports their difference. A correct
"wait then react" dataset has motion onset >= audio onset (diff >= 0).
"""
from __future__ import annotations

import argparse
import glob
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-dir", required=True)
    ap.add_argument("--state-threshold", type=float, default=0.05,
                    help="max-abs joint-state deviation from frame0 = 'moving'")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.lerobot_dir}/data/**/*.parquet",
                             recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {args.lerobot_dir}/data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    audio, motion, diff = [], [], []
    n_audio_from0 = 0
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        cls = np.stack(g["observation.audio.class_id"].values)   # (T, K)
        st = np.stack(g["observation.state"].values)             # (T, D)
        active = (cls != -1).any(axis=1)
        if not active.any():
            continue
        a = int(np.argmax(active))
        if active[0]:
            n_audio_from0 += 1
        dev = np.max(np.abs(st - st[0]), axis=1)
        if not (dev > args.state_threshold).any():
            continue
        m = int(np.argmax(dev > args.state_threshold))
        audio.append(a / args.fps)
        motion.append(m / args.fps)
        diff.append((m - a) / args.fps)

    audio = np.array(audio); motion = np.array(motion); diff = np.array(diff)
    n = len(diff)
    print(f"episodes analysed: {n}  (audio active from frame0 in "
          f"{n_audio_from0})")
    print(f"audio onset  s: mean {audio.mean():.2f}  min {audio.min():.2f}  "
          f"max {audio.max():.2f}")
    print(f"motion onset s: mean {motion.mean():.2f}  min {motion.min():.2f}  "
          f"max {motion.max():.2f}")
    print(f"motion-audio s: mean {diff.mean():+.2f}  median "
          f"{np.median(diff):+.2f}  (>=0 = waited for chime)")
    print(f"fraction moved AFTER audio onset: {float((diff >= 0).mean()):.2f}")
    if diff.mean() >= 0:
        print("=> GOOD: demos wait for the chime then move.")
    else:
        print("=> BAD: demos move before the chime (data still not gated).")


if __name__ == "__main__":
    main()
