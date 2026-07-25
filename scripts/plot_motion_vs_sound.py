#!/usr/bin/env python3
"""Visualise robot-motion onset vs sound onset for the take_out_microwave_food
eval.

For each displacement threshold (default 1/2/5/10/20 cm) it computes, per
episode, the time the end-effector first travels that far from its rest pose
(motion onset), the chime onset time, and their difference (motion - sound).
It then renders:

  * motion_vs_sound_scatter.png  - per-threshold scatter of sound onset (x)
    vs motion onset (y) with the y=x line; points below the line moved before
    the sound.
  * motion_vs_sound_diff_hist.png - per-threshold histogram of the difference
    (motion - sound); <0 means the robot moved before the sound.
  * motion_vs_sound_onset_cdf.png - CDF of motion onset per threshold overlaid
    with the (threshold-independent) sound-onset CDF.

Source of truth is <eval_dir>/motion_traces/*.json (written by eval with
--save-motion-trace), so any threshold can be derived without re-running.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _crossing_sec(trace, thr, step_dt):
    """First time (s) the displacement exceeds `thr`, or None."""
    for step, disp in trace:
        if disp > thr:
            return step * step_dt
    return None


def load_episodes(eval_dir: Path):
    trace_files = sorted(glob.glob(str(eval_dir / "motion_traces" / "*.json")))
    eps = []
    for tf in trace_files:
        with open(tf) as f:
            d = json.load(f)
        eps.append(d)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True,
                    help="Eval output dir containing motion_traces/.")
    ap.add_argument("--thresholds", default="0.01,0.02,0.05,0.1,0.2",
                    help="Comma-separated displacement thresholds in metres.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write PNGs (default: <eval-dir>).")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thrs = [float(t) for t in args.thresholds.split(",") if t.strip()]

    eps = load_episodes(eval_dir)
    if not eps:
        raise SystemExit(f"no motion traces under {eval_dir}/motion_traces/")

    sound = np.array([e["sound_onset_sec"] for e in eps], dtype=float)
    # motion[thr] = array of motion-onset secs (nan if never crossed)
    motion = {}
    diff = {}
    for thr in thrs:
        m = []
        for e in eps:
            c = _crossing_sec(e["trace"], thr, e["step_dt_sec"])
            m.append(c if c is not None else np.nan)
        m = np.array(m, dtype=float)
        motion[thr] = m
        diff[thr] = m - sound

    n = len(eps)
    cm_labels = {0.01: "1 cm", 0.02: "2 cm", 0.05: "5 cm",
                 0.1: "10 cm", 0.2: "20 cm"}

    def lbl(thr):
        return cm_labels.get(thr, f"{thr*100:g} cm")

    lim = float(np.nanmax([np.nanmax(motion[t]) for t in thrs] + [sound.max()])) * 1.05

    # ---- Figure 1: scatter sound vs motion, per threshold ----------------
    fig, axes = plt.subplots(1, len(thrs), figsize=(4 * len(thrs), 4.2),
                             sharex=True, sharey=True)
    if len(thrs) == 1:
        axes = [axes]
    for ax, thr in zip(axes, thrs):
        m = motion[thr]
        valid = ~np.isnan(m)
        before = valid & (m < sound)
        after = valid & (m >= sound)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x")
        ax.scatter(sound[before], m[before], c="tab:red", s=32,
                   label="moved before sound", zorder=3)
        ax.scatter(sound[after], m[after], c="tab:green", s=32,
                   label="moved after sound", zorder=3)
        n_after = int(after.sum())
        n_moved = int(valid.sum())
        ax.set_title(f"{lbl(thr)}  (after: {n_after}/{n_moved})")
        ax.set_xlabel("sound onset (s)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("motion onset (s)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Motion onset vs sound onset  (n={n} episodes)", y=1.02)
    fig.tight_layout()
    p1 = out_dir / "motion_vs_sound_scatter.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: histogram of difference, per threshold ----------------
    all_d = np.concatenate([diff[t][~np.isnan(diff[t])] for t in thrs])
    dmin, dmax = float(all_d.min()), float(all_d.max())
    bins = np.linspace(dmin, dmax, 16)
    fig, axes = plt.subplots(1, len(thrs), figsize=(4 * len(thrs), 4.2),
                             sharex=True, sharey=True)
    if len(thrs) == 1:
        axes = [axes]
    for ax, thr in zip(axes, thrs):
        d = diff[thr][~np.isnan(diff[thr])]
        ax.hist(d, bins=bins, color="tab:blue", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="k", ls="--", lw=1)
        med = float(np.median(d)) if len(d) else float("nan")
        mean = float(np.mean(d)) if len(d) else float("nan")
        ax.axvline(mean, color="tab:orange", ls="-", lw=1.5,
                   label=f"mean {mean:.1f}s")
        ax.set_title(f"{lbl(thr)}  (median {med:.1f}s)")
        ax.set_xlabel("motion - sound (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("episodes")
    fig.suptitle("Distribution of (motion onset - sound onset)  "
                 "[<0 = moved before sound]", y=1.02)
    fig.tight_layout()
    p2 = out_dir / "motion_vs_sound_diff_hist.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: CDF of motion onset per threshold + sound onset CDF ----
    fig, ax = plt.subplots(figsize=(7, 5))

    def cdf(vals):
        v = np.sort(vals[~np.isnan(vals)])
        y = np.arange(1, len(v) + 1) / len(v)
        return v, y

    for thr in thrs:
        v, y = cdf(motion[thr])
        ax.step(v, y, where="post", lw=1.8, label=f"motion {lbl(thr)}")
    sv, sy = cdf(sound)
    ax.step(sv, sy, where="post", lw=2.6, color="k", ls="--",
            label="sound onset")
    ax.set_xlabel("time since episode start (s)")
    ax.set_ylabel("cumulative fraction of episodes")
    ax.set_title(f"Onset CDFs: motion (by threshold) vs sound  (n={n})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p3 = out_dir / "motion_vs_sound_onset_cdf.png"
    fig.savefig(p3, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- console summary -------------------------------------------------
    print(f"episodes with traces: {n}")
    print(f"sound onset (s): mean {sound.mean():.2f}  "
          f"min {sound.min():.2f}  max {sound.max():.2f}")
    print(f"{'thr':>6} {'moved':>6} {'after_sound':>12} "
          f"{'mean_motion_s':>14} {'mean_diff_s':>12}")
    for thr in thrs:
        m = motion[thr]
        valid = ~np.isnan(m)
        d = diff[thr][valid]
        after = int((m[valid] >= sound[valid]).sum())
        print(f"{lbl(thr):>6} {int(valid.sum()):>6} "
              f"{after:>5}/{int(valid.sum()):<6} "
              f"{np.nanmean(m):>14.2f} {np.mean(d):>12.2f}")
    print(f"\n[saved] {p1}\n[saved] {p2}\n[saved] {p3}")


if __name__ == "__main__":
    main()
