#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ("select_radio", "select_radio_two", "select_radio_silent")
TASK_LABELS = {
    "select_radio": "one",
    "select_radio_two": "two",
    "select_radio_silent": "silent",
}


def _read_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def collect_rows(eval_root: Path, steps: list[int]) -> list[dict]:
    rows = []
    for step in steps:
        label = f"step{step // 1000:03d}k"
        for task in TASKS:
            summary_path = eval_root / f"{label}_{task}" / "eval_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            summary = _read_summary(summary_path)
            rows.append({
                "step": step,
                "step_k": step // 1000,
                "task": task,
                "task_label": TASK_LABELS[task],
                "n_episodes": summary.get("n_episodes"),
                "success_rate": summary.get("success_rate"),
                "intention_success_rate": summary.get("intention_success_rate"),
                "legacy_intention_success_rate": summary.get("legacy_intention_success_rate"),
                "exclusive_intention_success_rate": summary.get("exclusive_intention_success_rate"),
                "avg_progress": summary.get("avg_progress"),
                "ckpt": summary.get("ckpt"),
            })
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "step_k",
        "task",
        "task_label",
        "n_episodes",
        "success_rate",
        "intention_success_rate",
        "legacy_intention_success_rate",
        "exclusive_intention_success_rate",
        "avg_progress",
        "ckpt",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    metrics = [
        ("success_rate", "Success rate"),
        ("intention_success_rate", "Intention success rate"),
    ]
    colors = {
        "select_radio": "#2f6f9f",
        "select_radio_two": "#c05a2b",
        "select_radio_silent": "#3f8f5f",
    }
    for ax, (metric, title) in zip(axes, metrics):
        for task in TASKS:
            task_rows = [row for row in rows if row["task"] == task]
            xs = [row["step_k"] for row in task_rows]
            ys = [row[metric] for row in task_rows]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=TASK_LABELS[task],
                color=colors[task],
            )
        ax.set_title(title)
        ax.set_xlabel("Training step (k)")
        ax.set_xticks(sorted({row["step_k"] for row in rows}))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(title="Task", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("outputs/eval_smolvla_oracle_all_mixed_fixed_one_nl_azcanon_lora_40k_80k_120k_160k"),
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[40000, 80000, 120000, 160000])
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    rows = collect_rows(args.eval_root, args.steps)
    csv_path = args.csv or args.eval_root / "summary.csv"
    plot_path = args.plot or args.eval_root / "success_rates.png"
    write_csv(rows, csv_path)
    write_plot(rows, plot_path)
    print(f"[saved] {csv_path}")
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
