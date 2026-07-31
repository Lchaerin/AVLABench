#!/usr/bin/env python
"""Render candidate cam_0 / cam_2 placements for a find_hidden v4 dataset.

Why this exists
---------------
v3 put cam_0 straight overhead (`0 0.15 2.10`, looking down). That view says
almost nothing about *which drawer* the gripper is at, which is exactly the
discrimination the bottom-drawer slot needs. The proposal is to go back to
VLABench's stock table-side cameras — the ones `dataset_find_hidden_lerobot_src`
was generated with, before `camera_config.json` gained a find_hidden entry:

    cam_0 <- stock "right" (0.775, -0.856, 1.209)
    cam_2 <- stock "left" (-0.775, -0.856, 1.209)

...but raised a little and tilted down a little, so the drawer fronts are seen
from slightly above instead of edge-on.

The microphone (cam_1) is NOT touched: its pose fixes the audio labels, and v3's
(0, -1.05, 1.20) is what the current d' 16.4 / 16.9 separation was measured at.

Fairness
--------
All candidates are rendered from the *same* rollout. The oracle runs once; every
physics state is snapshotted; then for each candidate the camera is moved at
runtime (`physics.model.cam_pos/cam_quat`), the state is restored and the frame
re-rendered. Cameras do not affect physics, so the scenes are identical and any
visible difference is the camera alone.

Usage
-----
    MUJOCO_GL=egl python scripts/preview_find_hidden_cameras.py \
        --slots left_bottom right_top --out outputs/cam_candidates
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Stock VLABench table-side cameras (VLABench/assets/base/camera.xml).
STOCK_RIGHT = np.array([0.775, -0.856, 1.209])
STOCK_LEFT = np.array([-0.775, -0.856, 1.209])

# Candidate placements. `lift` raises the camera; `aim` is the world point it
# looks at, which is what actually sets the downward tilt; `out` pushes the
# camera away from the scene centre in x/y to widen the framing.
CANDIDATES = {
    "A_minimal": dict(lift=0.10, aim=(0.0, 0.10, 0.98), out=1.00, fovy=45.0),
    "B_mild":    dict(lift=0.20, aim=(0.0, 0.10, 0.96), out=1.00, fovy=45.0),
    "C_moderate":dict(lift=0.30, aim=(0.0, 0.10, 0.94), out=1.00, fovy=45.0),
    "D_wide":    dict(lift=0.20, aim=(0.0, 0.10, 0.96), out=1.15, fovy=55.0),
}


def look_at_quat(pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo camera orientation looking from `pos` at `target`.

    MuJoCo cameras look down their own -Z, with +X right and +Y up. Returns
    (wxyz quaternion, xyaxes) — the latter is what camera_config.json stores.
    """
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    z_cam = -fwd                                   # camera looks along -Z
    x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.column_stack([x_cam, y_cam, z_cam])
    # rotation matrix -> wxyz quaternion
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:                                   # degenerate; fall back to scipy
        from scipy.spatial.transform import Rotation as Rot
        x, y, z, w = Rot.from_matrix(R).as_quat()
        return np.array([w, x, y, z]), np.concatenate([x_cam, y_cam])
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z]), np.concatenate([x_cam, y_cam])


def candidate_poses(spec: dict) -> dict[str, dict]:
    """cam_0 (from stock right) and cam_2 (from stock left) for one candidate."""
    out = {}
    for cam_idx, stock in ((0, STOCK_RIGHT), (2, STOCK_LEFT)):
        pos = stock.copy()
        pos[0] *= spec["out"]
        pos[1] *= spec["out"]
        pos[2] += spec["lift"]
        quat, xyaxes = look_at_quat(pos, spec["aim"])
        tilt = np.degrees(np.arcsin(
            (pos[2] - spec["aim"][2]) / np.linalg.norm(pos - np.asarray(spec["aim"], float))))
        out[cam_idx] = dict(pos=pos, quat=quat, xyaxes=xyaxes,
                            fovy=spec["fovy"], tilt_down_deg=tilt)
    return out


def cam_name_to_index(physics) -> dict[str, int]:
    names = {}
    for i in range(physics.model.ncam):
        names[physics.model.id2name(i, "camera")] = i
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", nargs="+", default=["left_bottom", "right_top"])
    ap.add_argument("--out", default="outputs/cam_candidates")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--n-frames", type=int, default=4)
    args = ap.parse_args()

    import cv2
    # Importing these packages is what populates the robot/task registries;
    # without them load_env raises KeyError('franka').
    import VLABench.robots  # noqa: F401
    import VLABench.tasks   # noqa: F401
    from VLABench.envs import load_env
    from VLABench.tasks.hierarchical_tasks.primitive.find_hidden_object_open_series import (
        close_all_drawers,
    )

    os.makedirs(args.out, exist_ok=True)

    for cand, spec in CANDIDATES.items():
        poses = candidate_poses(spec)
        print(f"{cand}: lift {spec['lift']:+.2f} m, fovy {spec['fovy']:.0f}, "
              f"tilt down {poses[0]['tilt_down_deg']:.1f} deg")
        for ci in (0, 2):
            p = poses[ci]["pos"]
            print(f"    cam_{ci}: pos {np.round(p,3).tolist()}  "
                  f"xyaxes {' '.join(f'{v:.3f}' for v in poses[ci]['xyaxes'])}")

    for slot in args.slots:
        os.environ["VLABENCH_HIDDEN_SLOT_LABEL"] = slot
        env = load_env("find_hidden_object_open", robot="franka")
        env.reset()
        close_all_drawers(env)

        physics = env.physics
        cams = cam_name_to_index(physics)
        print(f"\n[{slot}] cameras in model: {cams}")

        # --- run the oracle once, snapshotting physics after every step -------
        states = [(physics.data.qpos.copy(), physics.data.qvel.copy())]
        orig_step = env.step

        def recording_step(action, _orig=orig_step, _acc=states):
            ts = _orig(action)
            _acc.append((env.physics.data.qpos.copy(), env.physics.data.qvel.copy()))
            return ts

        env.step = recording_step
        try:
            for skill in env.get_expert_skill_sequence():
                _obs, _wp, _stage_ok, task_ok = skill(env)
                if task_ok:
                    break
        finally:
            env.step = orig_step
        print(f"[{slot}] rollout captured {len(states)} states")

        # Frames to show: start, two thirds through the approach, and the end
        # (drawer open) — the moment that tells you whether the view resolves
        # *which* drawer moved.
        n = len(states)
        picks = [0] + [int(n * f) for f in (0.55, 0.8)] + [n - 1]
        picks = sorted(set(min(max(p, 0), n - 1) for p in picks))[: args.n_frames]

        for cand, spec in CANDIDATES.items():
            poses = candidate_poses(spec)
            for ci, pose in poses.items():
                physics.model.cam_pos[ci] = pose["pos"]
                physics.model.cam_quat[ci] = pose["quat"]
                physics.model.cam_fovy[ci] = pose["fovy"]

            rows = []
            for ci in (0, 2):
                imgs = []
                for p in picks:
                    qpos, qvel = states[p]
                    physics.data.qpos[:] = qpos
                    physics.data.qvel[:] = qvel
                    physics.forward()
                    imgs.append(physics.render(camera_id=ci,
                                               height=args.size, width=args.size))
                rows.append(np.concatenate(imgs, axis=1))
            sheet = np.concatenate(rows, axis=0)
            path = os.path.join(args.out, f"{slot}__{cand}.png")
            cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            print(f"  wrote {path}")

        env.close()

    # camera_config.json snippets, ready to paste
    print("\n=== camera_config.json snippets (cam_1 unchanged) ===")
    for cand, spec in CANDIDATES.items():
        poses = candidate_poses(spec)
        print(f"\n-- {cand}")
        for ci in (0, 2):
            p, x = poses[ci]["pos"], poses[ci]["xyaxes"]
            print(f'  "{ci}": {{"pos": "{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}", '
                  f'"xyaxes": "{" ".join(f"{v:.4f}" for v in x)}", '
                  f'"fovy": "{spec["fovy"]:.0f}"}},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
