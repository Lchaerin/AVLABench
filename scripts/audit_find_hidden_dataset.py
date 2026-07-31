#!/usr/bin/env python
"""Audit a find_hidden dataset for the defects that made the v2 set unusable.

Run after generation/conversion. Checks, in order of how badly each one bites:

  1. Degenerate episodes — a drawer that opened on its own satisfies the success
     condition before the expert moves, so the episode is a ~2 s stay-still clip.
     Detected by frame count and end-effector travel.
  2. Object placement — the hidden object must sit in the drawer its slot label
     names. If it fell to another level the stored audio elevation points at the
     wrong drawer.
  3. Audio cue separability — d' on the (u, v) features the SlotEncoder consumes,
     which is what actually decides whether the task is solvable from sound.
  4. Slot balance.
  5. LeRobot base frame — the state's first frame must match what eval's
     get_robot_frame_position() produces, or the policy sees a shifted world (the
     bug that pinned success at 0 %; see KISTI_pi0_STATUS.md).

Exits non-zero if any hard check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.audio.projection import CameraIntrinsics, doa_to_uv  # noqa: E402

# Thresholds mirror the generator's gates so the audit and the gate agree.
from VLABench.tasks.hierarchical_tasks.primitive.find_hidden_object_open_series import (  # noqa: E402
    ALL_LEVEL_LOCAL_Z, MIN_EE_TRAVEL_M, MIN_EPISODE_FRAMES, SETTLED_LOCAL_Z,
    OBJECT_MAX_DZ_ERROR,
)

# Eval's env.get_robot_frame_position() for this scene. The converter shifts
# state/action into this frame; the first recorded frame is the robot at rest.
EXPECTED_FIRST_STATE = np.array([0.0004, 0.2297, 0.4319])
FIRST_STATE_TOL_M = 0.01
# How many episodes to sample for the base-frame check (one parquet each).
BASE_FRAME_SAMPLE_EPISODES = 80


def _open_h5(path: str):
    """Read-only open that tolerates a generator still writing in the directory
    (h5py opens for append, which otherwise makes the whole dir unreadable)."""
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:            # h5py too old for the `locking` kwarg
        return h5py.File(path, "r")


def load_episodes(src_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(src_dir, "data_*.hdf5")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not files:
        raise SystemExit(f"no data_*.hdf5 under {src_dir}")
    rows, unreadable = [], []
    for path in files:
        idx = int(os.path.basename(path).split("_")[1].split(".")[0])
        try:
            f = _open_h5(path)
        except OSError as exc:
            # A truncated / half-written file. Report it rather than aborting the
            # whole audit, so one bad episode does not hide the other findings.
            unreadable.append((os.path.basename(path), str(exc).split("(")[0].strip()))
            continue
        with f:
            grp = f["data"][list(f["data"].keys())[0]]
            act = grp["action"][...]
            ec = json.loads(grp["meta_info/episode_config"][()].decode())
            oa = json.loads(grp["meta_info/oracle_audio"][()].decode())
            ee0 = grp["observation/ee_state"][0][:3]
            eeL = grp["observation/ee_state"][-1][:3]
        cond = ec["task"]["conditions"]["drawer_open"]
        comps = {c["name"]: c for c in ec["task"]["components"]}
        rows.append(dict(
            idx=idx,
            n=int(act.shape[0]),
            side=cond["container"].replace("cabinet_", ""),
            elev=cond["elevation"],
            obj=ec["task"]["target_entity"],
            cab=np.array(comps[cond["container"]]["position"], dtype=float),
            travel=float(np.linalg.norm(np.asarray(eeL) - np.asarray(ee0))),
            oracle=oa,
        ))
    return rows, unreadable


def dprime(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.sqrt(0.5 * (a.var() + b.var()))
    return float((b.mean() - a.mean()) / denom) if denom > 0 else float("nan")


def audit_source(src_dir: str) -> list[str]:
    rows, unreadable = load_episodes(src_dir)
    failures = []
    print(f"\n=== source HDF5: {len(rows)} episodes in {src_dir} ===")
    if unreadable:
        failures.append(f"{len(unreadable)} unreadable HDF5 file(s): "
                        + ", ".join(f"{n} ({why})" for n, why in unreadable[:5]))
    if not rows:
        return failures

    # --- 1. degenerate episodes -------------------------------------------
    ns = np.array([r["n"] for r in rows])
    tr = np.array([r["travel"] for r in rows])
    short = [r for r in rows if r["n"] < MIN_EPISODE_FRAMES or r["travel"] < MIN_EE_TRAVEL_M]
    print(f"frames  min {ns.min()} p1 {np.percentile(ns,1):.0f} med {np.median(ns):.0f} "
          f"max {ns.max()}")
    print(f"travel  min {tr.min():.3f} p1 {np.percentile(tr,1):.3f} "
          f"med {np.median(tr):.3f} max {tr.max():.3f} m")
    if short:
        failures.append(f"{len(short)} degenerate episodes (too short / barely moved): "
                        + ", ".join(f"ep{r['idx']}(n={r['n']},{r['travel']:.2f}m)"
                                    for r in short[:10]))
    else:
        print(f"OK  no episode under {MIN_EPISODE_FRAMES} frames or "
              f"{MIN_EE_TRAVEL_M} m of travel")

    # --- 2. object placement ---------------------------------------------
    misplaced, wrong_level = [], []
    for r in rows:
        xpos = np.array(r["oracle"]["active_sources"][0]["xpos"], dtype=float)
        dz = float(xpos[2] - r["cab"][2])
        if abs(dz - SETTLED_LOCAL_Z[r["elev"]]) > OBJECT_MAX_DZ_ERROR:
            misplaced.append((r["idx"], r["elev"], dz))
        nearest = min(ALL_LEVEL_LOCAL_Z, key=lambda k: abs(dz - ALL_LEVEL_LOCAL_Z[k]))
        if nearest != r["elev"]:
            wrong_level.append((r["idx"], r["elev"], nearest, dz))
    if wrong_level:
        failures.append(f"{len(wrong_level)} episodes whose object sits at a different "
                        f"drawer level than its label: "
                        + ", ".join(f"ep{i}({lab}->{near},dz={dz:+.3f})"
                                    for i, lab, near, dz in wrong_level[:10]))
    else:
        print("OK  every object is nearest the drawer level its slot label names")
    if misplaced:
        print(f"note {len(misplaced)} episodes exceed the "
              f"{OBJECT_MAX_DZ_ERROR} m height tolerance but stay on the right level")

    # --- 3. audio separability -------------------------------------------
    cam_fovy = rows[0]["oracle"]["cam_fovy_deg"]
    mic_cams = {r["oracle"]["cam_id"] for r in rows}
    mic_pos = np.round(rows[0]["oracle"]["cam_pos"], 3).tolist()
    print(f"\nmic: camera {sorted(mic_cams)} at {mic_pos}, fovy {cam_fovy:.0f}")
    if len(mic_cams) != 1:
        failures.append(f"episodes disagree on the mic camera: {sorted(mic_cams)}")

    K = CameraIntrinsics.from_fovy(cam_fovy, 224, 224)
    feats = defaultdict(list)
    offscreen = 0
    for r in rows:
        src = r["oracle"]["active_sources"][0]
        uv = doa_to_uv(src["az_deg"], src["el_deg"], K, (224, 224), convention="sled")
        if uv is None:
            offscreen += 1
            continue
        u, v = uv
        for key, val in (("az", src["az_deg"]), ("el", src["el_deg"]), ("u", u), ("v", v)):
            feats[(r["side"], r["elev"], key)].append(val)
            feats[(r["elev"], key)].append(val)
            feats[(r["side"], key)].append(val)
    print(f"{'slot':<14} {'n':>5} {'az':>15} {'el':>15} {'u':>14} {'v':>14}")
    for side in ("left", "right"):
        for elev in ("top", "bottom"):
            n = len(feats[(side, elev, "az")])
            if n == 0:
                print(f"{side + '_' + elev:<14} {0:>5}  (absent)")
                continue
            cells = []
            for key, width in (("az", 15), ("el", 15), ("u", 14), ("v", 14)):
                a = np.array(feats[(side, elev, key)])
                cells.append(f"{a.mean():+.3f}±{a.std():.3f}".rjust(width))
            print(f"{side + '_' + elev:<14} {n:>5} {' '.join(cells)}")

    print(f"offscreen {offscreen}/{len(rows)}")
    if offscreen:
        failures.append(f"{offscreen} episodes project off-screen (audio slot masked out)")

    # d' only means something when both classes of the contrast are present.
    for name, a_key, b_key, feat, floor in (
        ("azimuth (u, left vs right)", ("left", "u"), ("right", "u"), "u", 5.0),
        ("elevation (v, top vs bottom)", ("bottom", "v"), ("top", "v"), "v", 4.0),
    ):
        if not feats[a_key] or not feats[b_key]:
            print(f"d'  {name}: skipped (only one side present in this set)")
            continue
        d = dprime(feats[a_key], feats[b_key])
        print(f"d'  {name} {d:+.2f}")
        if abs(d) < floor:
            failures.append(f"{feat} cue too weak: |d'| {abs(d):.2f} < {floor}")

    # --- 4. slot balance --------------------------------------------------
    counts = defaultdict(int)
    for r in rows:
        counts[f"{r['side']}_{r['elev']}"] += 1
    print(f"\nslot balance: {dict(sorted(counts.items()))}")
    if counts and (max(counts.values()) - min(counts.values())) > 0:
        print("note slots are not exactly balanced")
    if len(counts) != 4:
        failures.append(f"expected 4 slots, found {sorted(counts)}")

    # --- 5. target-object mix ---------------------------------------------
    # The object's resting height depends on which object it is (measured on the
    # v2 set: per-object median dz spans 0.045-0.074 in the bottom drawer), and
    # the placement gate keys off height — so a too-tight gate could quietly
    # skew the object mix. The generator draws uniformly from 3 seen objects, so
    # a large deviation from 1/3 means the gate is filtering by object identity.
    obj_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        obj_counts[r["elev"]][r["obj"]] += 1
    for elev in sorted(obj_counts):
        tot = sum(obj_counts[elev].values())
        shares = {o: n / tot for o, n in sorted(obj_counts[elev].items())}
        print(f"object mix ({elev}, n={tot}): "
              + ", ".join(f"{o} {100 * s:.0f}%" for o, s in shares.items()))
        if tot >= 100:
            expected = 1.0 / max(len(shares), 1)
            # ~4 sigma on a binomial share at this n; a real gate-induced skew is
            # far larger than sampling noise.
            tol = 4 * np.sqrt(expected * (1 - expected) / tot)
            skewed = {o: s for o, s in shares.items() if abs(s - expected) > tol}
            if skewed:
                # A WARNING, not a failure. The gate does filter by object height
                # — `boxed_food` is the least stable object in the top drawer, so
                # more of its episodes end up at the middle-drawer level and are
                # (correctly) rejected. For find_hidden_object_open that changes
                # nothing the policy can see: the object stays hidden for the
                # whole approach and only rides out in the last ~10-20 frames,
                # after the drawer has been chosen; measured on the 880-episode
                # v3 set the mix shifts the top class's mean elevation by 0.08
                # deg against a 12.4 deg top/bottom separation, and leaves energy
                # and sound class untouched (every object spans 33-36 of the 36
                # classes). It WOULD matter for a task that has to manipulate the
                # object — e.g. the composite find_hidden_object retrieve variant.
                print(
                    f"WARN object mix skewed for {elev} (expected ~{100*expected:.0f}% "
                    f"each, tol ±{100*tol:.0f}%): "
                    + ", ".join(f"{o} {100*s:.0f}%" for o, s in skewed.items())
                    + "\n     The placement gate filters by object height. Harmless "
                      "here (the object is never observed before the drawer choice), "
                      "but re-check this if the task ever requires grasping it."
                )
    return failures


def audit_lerobot(lerobot_dir: str) -> list[str]:
    """Check the converted set's base frame — the v1/v2 killer bug."""
    failures = []
    print(f"\n=== LeRobot: {lerobot_dir} ===")
    parquets = sorted(glob.glob(os.path.join(lerobot_dir, "data", "**", "*.parquet"),
                                recursive=True))
    if not parquets:
        return [f"no parquet files under {lerobot_dir}/data"]
    try:
        import pandas as pd
    except ImportError:
        print("note pandas unavailable; skipping base-frame check")
        return failures

    info_path = os.path.join(lerobot_dir, "meta", "info.json")
    if os.path.exists(info_path):
        info = json.load(open(info_path))
        print(f"episodes {info.get('total_episodes')}  frames {info.get('total_frames')}  "
              f"fps {info.get('fps')}")
        print(f"features: {sorted(info.get('features', {}))}")

    # LeRobot v2.1 writes ONE parquet per episode, so a single file is a single
    # episode. The robot's rest pose is bimodal (measured on v2: 83% at
    # base-frame z=0.4315, 17% at 0.4687 — a property of env.reset()'s settle
    # that eval samples too), so one episode is a coin flip. Take the MEDIAN over
    # many episodes: it lands on the dominant mode and still moves under a
    # systematic shift, which is what this check is for.
    firsts = []
    for path in parquets[:BASE_FRAME_SAMPLE_EPISODES]:
        df = pd.read_parquet(path)
        for _, ep in df.groupby("episode_index"):
            firsts.append(np.asarray(
                ep.sort_values("frame_index").iloc[0]["observation.state"],
                dtype=float)[:3])
    firsts = np.stack(firsts)
    med_first = np.median(firsts, axis=0)
    err = np.abs(med_first - EXPECTED_FIRST_STATE)
    print(f"first-frame state, median over {len(firsts)} episodes: "
          f"{np.round(med_first, 4).tolist()}")
    print(f"eval's measured rest pose:                    "
          f"{EXPECTED_FIRST_STATE.tolist()}")
    print(f"error: {np.round(err, 4).tolist()} m   "
          f"(per-axis spread p5..p95 "
          f"{np.round(np.percentile(firsts, 5, axis=0), 4).tolist()}..."
          f"{np.round(np.percentile(firsts, 95, axis=0), 4).tolist()})")
    if err.max() > FIRST_STATE_TOL_M:
        failures.append(
            f"base-frame mismatch: median first-frame state is off by {err.max():.3f} m "
            f"(> {FIRST_STATE_TOL_M} m) from what eval measures at t=0. Either the "
            f"converter's ROBOT_FRAME_POS shift did not apply (check --use-real-state), "
            f"or extra physics steps ran before recording started and let the arm sag "
            f"(see close_all_drawers / _restore_robot_pose). Both are silent train/eval "
            f"divergences — see KISTI_pi0_STATUS.md."
        )
    else:
        print("OK  base frame matches eval (KISTI_pi0_STATUS.md frame fix is in effect)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True, help="…/find_hidden_object_open with data_*.hdf5")
    ap.add_argument("--lerobot-dir", default=None, help="converted LeRobot dataset root")
    args = ap.parse_args()

    failures = audit_source(args.src_dir)
    if args.lerobot_dir and os.path.isdir(args.lerobot_dir):
        failures += audit_lerobot(args.lerobot_dir)

    print()
    if failures:
        print(f"FAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("PASS — all checks clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
