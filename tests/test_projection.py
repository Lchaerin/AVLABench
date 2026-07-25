"""Unit tests for M2 DoA→(u,v) projection (seld_vla_implementation_spec.md §4)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio.projection import CameraIntrinsics, doa_to_uv

W, H = 224, 224
K = CameraIntrinsics.from_fovy(60.0, W, H)


def test_center():
    # az=0, el=0 → principal point (image centre).
    u, v = doa_to_uv(0.0, 0.0, K, (W, H))
    assert u == pytest.approx(0.5, abs=1e-6)
    assert v == pytest.approx(0.5, abs=1e-6)


def test_dcase_left_positive():
    # DCASE az=+15° is LEFT → u < 0.5.
    u, v = doa_to_uv(15.0, 0.0, K, (W, H), convention="dcase")
    assert u < 0.5


def test_elevation_up():
    # el=+20° (up) → v < 0.5 (upper image).
    u, v = doa_to_uv(0.0, 20.0, K, (W, H))
    assert v < 0.5


def test_sled_right_positive():
    # SLED az=+15° is RIGHT → u > 0.5. Same angle, opposite side vs DCASE.
    u, _ = doa_to_uv(15.0, 0.0, K, (W, H), convention="sled")
    assert u > 0.5
    # sled(+15) and dcase(-15) must agree.
    u_d, _ = doa_to_uv(-15.0, 0.0, K, (W, H), convention="dcase")
    assert u == pytest.approx(u_d, abs=1e-9)


def test_behind_camera_is_none():
    assert doa_to_uv(180.0, 0.0, K, (W, H)) is None


def test_offscreen_is_none():
    # 80° azimuth with a 60° vertical FOV falls outside the frame.
    assert doa_to_uv(80.0, 0.0, K, (W, H)) is None


def test_rotation_offset_identity_noop():
    u0, v0 = doa_to_uv(10.0, 5.0, K, (W, H))
    u1, v1 = doa_to_uv(10.0, 5.0, K, (W, H), R_mic2cam=np.eye(3))
    assert u0 == pytest.approx(u1) and v0 == pytest.approx(v1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
