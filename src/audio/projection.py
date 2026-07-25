"""M2 — DoA (azimuth/elevation) → fixed-camera image coordinate (u, v) projection.

Implements the spec's `doa_to_uv` (seld_vla_implementation_spec.md §4). The
sounding object's direction, expressed as (azimuth, elevation) relative to the
listener, is projected onto the fixed 3rd-person camera that shares the mic's
pose so the policy receives a *visually grounded* slot position rather than a
raw angle.

Coordinate conventions (the spec's #1 pitfall — see the unit tests)
------------------------------------------------------------------
Two azimuth sign conventions appear in this repo:

* ``dcase``  — DCASE standard used verbatim in the spec: x forward, y **left**,
  z up. Azimuth is **left-positive** (counter-clockwise from front).
* ``sled``   — what this project actually *stores* in the dataset
  (``observation.audio.azimuth_deg``). ``oracle_sled.gt_az_el`` negates the HRTF
  azimuth so that **right is positive** (``az_sled = -az_hrtf``). This matches
  the real SLED model's ``arctan2(dy, dx)`` output.

Because the dataset stores SLED azimuth, callers projecting stored events must
pass ``convention="sled"`` (or pre-negate). ``dcase`` is kept so the spec's
reference math can be unit-tested exactly as written.

The camera is OpenCV-style: X right, Y down, Z forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels."""
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fovy(fovy_deg: float, width: int, height: int) -> "CameraIntrinsics":
        """Build intrinsics from a MuJoCo camera's vertical FOV.

        MuJoCo cameras specify ``fovy`` (vertical field of view, degrees).
        Pixels are square, so fx == fy, and the principal point is the image
        centre. ``fy = (H/2) / tan(fovy/2)``.
        """
        fov = radians(float(fovy_deg))
        fy = (height / 2.0) / tan(fov / 2.0)
        fx = fy  # square pixels
        return CameraIntrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)


def doa_to_uv(
    az_deg: float,
    el_deg: float,
    K: CameraIntrinsics,
    wh: tuple[int, int],
    R_mic2cam: Optional[np.ndarray] = None,
    convention: str = "dcase",
) -> Optional[tuple[float, float]]:
    """Project a direction-of-arrival onto normalised image coordinates.

    Parameters
    ----------
    az_deg, el_deg : direction of arrival, degrees.
    K              : camera intrinsics (pixels).
    wh             : (width, height) of the image, pixels.
    R_mic2cam      : optional 3x3 rotation correcting a small mic↔cam pose
                     offset (rotation only; far-field so translation ignored).
    convention     : "dcase" (left-positive az, spec reference) or "sled"
                     (right-positive az, what this repo stores).

    Returns
    -------
    (u, v) normalised to [0, 1), or ``None`` if the source is behind the
    camera or outside the field of view (``offscreen_policy=drop``).
    """
    if convention not in ("dcase", "sled"):
        raise ValueError(f"convention must be 'dcase' or 'sled', got {convention!r}")
    az = radians(float(az_deg))
    if convention == "sled":
        # SLED azimuth is right-positive; DCASE is left-positive. Negate to
        # reuse the DCASE direction-vector math below.
        az = -az
    el = radians(float(el_deg))

    # DCASE frame direction vector: x forward, y left, z up.
    d_f = np.array([cos(el) * cos(az), cos(el) * sin(az), sin(el)], dtype=float)
    # DCASE → OpenCV camera: X = -y_left (right), Y = -z_up (down), Z = x_fwd.
    d_c = np.array([-d_f[1], -d_f[2], d_f[0]], dtype=float)
    if R_mic2cam is not None:
        d_c = np.asarray(R_mic2cam, dtype=float).reshape(3, 3) @ d_c

    if d_c[2] <= 1e-6:
        return None  # behind the camera

    u = K.fx * d_c[0] / d_c[2] + K.cx
    v = K.fy * d_c[1] / d_c[2] + K.cy
    W, H = wh
    if not (0.0 <= u < W and 0.0 <= v < H):
        return None  # outside the field of view
    return (u / W, v / H)
