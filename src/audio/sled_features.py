"""Convert sled_predictions JSON (per-frame DOA + class) to dense arrays."""
from __future__ import annotations

import json
from typing import Tuple

import numpy as np


EMPTY_CLASS = -1


def parse_sled_json(sled_json: str | dict, n_frames: int, top_k: int = 3
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert SLED JSON to dense per-frame arrays.

    Returns:
        class_id      [n_frames, top_k]  int32 (-1 if empty)
        azimuth_deg   [n_frames, top_k]  float32
        elevation_deg [n_frames, top_k]  float32
        confidence    [n_frames, top_k]  float32
    """
    if isinstance(sled_json, str):
        sled = json.loads(sled_json)
    else:
        sled = sled_json

    class_id = np.full((n_frames, top_k), EMPTY_CLASS, dtype=np.int32)
    azim = np.zeros((n_frames, top_k), dtype=np.float32)
    elev = np.zeros((n_frames, top_k), dtype=np.float32)
    conf = np.zeros((n_frames, top_k), dtype=np.float32)

    frames = sled.get("frames", []) or []
    for entry in frames:
        i = int(entry["frame_idx"])
        if i < 0 or i >= n_frames:
            continue
        preds = entry.get("predictions", []) or []
        # Sort by confidence (most-confident first), keep top_k
        preds = sorted(preds, key=lambda p: -float(p.get("confidence", 0.0)))[:top_k]
        for k, p in enumerate(preds):
            class_id[i, k] = int(p.get("class_id", EMPTY_CLASS))
            azim[i, k] = float(p.get("azimuth_deg", 0.0))
            elev[i, k] = float(p.get("elevation_deg", 0.0))
            conf[i, k] = float(p.get("confidence", 0.0))

    return class_id, azim, elev, conf


def temporal_smooth_topk(class_id: np.ndarray, azim: np.ndarray,
                         elev: np.ndarray, conf: np.ndarray,
                         lookback_frames: int = 25
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For frames where SLED returned no events, carry forward the most recent
    non-empty prediction (within `lookback_frames`). This stabilises the audio
    signal during brief drop-outs without leaking future information."""
    T, K = class_id.shape
    out_cid  = class_id.copy()
    out_az   = azim.copy()
    out_el   = elev.copy()
    out_conf = conf.copy()
    last_idx = -1
    for t in range(T):
        if (class_id[t] != EMPTY_CLASS).any():
            last_idx = t
            continue
        if last_idx >= 0 and (t - last_idx) <= lookback_frames:
            out_cid[t]  = class_id[last_idx]
            out_az[t]   = azim[last_idx]
            out_el[t]   = elev[last_idx]
            # Decay confidence linearly with age
            decay = 1.0 - (t - last_idx) / float(lookback_frames + 1)
            out_conf[t] = conf[last_idx] * max(decay, 0.0)
    return out_cid, out_az, out_el, out_conf
