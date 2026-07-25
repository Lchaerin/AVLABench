"""
SLEDOverlay — integrates the SLED v3 sound-source localisation model
with the LIBERO multi-camera renderer.

Background thread runs SLEDv3 inference on the most recent binaural audio
from a BinauralAudioEngine (44 100 Hz) after resampling to 48 000 Hz.
The estimated DOA unit-vector is projected onto the agentview camera frame
and rendered as a coloured crosshair + confidence label.

Coordinate convention for SLED doa_vec [dx, dy, dz]:
  x = forward  (camera optical axis)
  y = right     (camera horizontal)
  z = up        (camera vertical)

Pinhole projection onto image (u right, v down):
  f  = cam_h / (2 * tan(fovy/2))
  u  = cx + f * (dy / dx)
  v  = cy - f * (dz / dx)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from scipy.signal import resample_poly


def _build_id_to_label(class_map_path: str | None) -> dict[int, str]:
    """Build {class_id: label} from a class_map.json file.

    Keys in the JSON are ``"ClassName/filename.wav"``; values are integer ids.
    """
    if not class_map_path or not os.path.exists(class_map_path):
        return {}
    with open(class_map_path) as f:
        cm = json.load(f)
    id_to_label: dict[int, str] = {}
    for key, cid in cm.items():
        label = key.split("/")[0]          # "ClassName"
        if cid not in id_to_label:
            id_to_label[cid] = label
    return id_to_label

# ── SLED root path ────────────────────────────────────────────────────────────
_CROSSCORR_ROOT = os.path.join(os.path.dirname(__file__), "../../../crossCorr")
_CROSSCORR_ROOT = os.path.normpath(_CROSSCORR_ROOT)
if _CROSSCORR_ROOT not in sys.path:
    sys.path.insert(0, _CROSSCORR_ROOT)

from sled.model.sled import SLEDv3


class SLEDOverlay:
    """
    Loads a SLEDv3 checkpoint and runs inference in a background thread.

    Parameters
    ----------
    ckpt_path      : path to sled_best.pt
    sofa_path      : HRTF SOFA file (same as BinauralAudioEngine)
    class_map_path : optional path to class_map.json for human-readable labels
    native_sr      : audio engine sample rate (Hz), default 44 100
    conf_thresh    : minimum slot confidence to draw a marker
    infer_hz       : target inference rate (Hz)
    torch_device   : 'cuda' or 'cpu'
    """

    SLED_SR       = 48_000
    WINDOW_FRAMES = 48               # 48 × 960 = 46 080 samples @ 48 kHz
    HOP_SAMPLES   = 960
    FOVY_DEG      = 45.0             # agentview vertical FOV (MuJoCo default)

    # BGR colours for slot 0, 1, 2
    _COLORS = [
        (0x3c, 0x4c, 0xe7),
        (0x71, 0xcc, 0x2e),
        (0xdb, 0x98, 0x34),
    ]

    def __init__(
        self,
        ckpt_path:      str,
        sofa_path:      str,
        class_map_path: str | None = None,
        native_sr:      int        = 44_100,
        conf_thresh:    float      = 0.30,
        infer_hz:       float      = 5.0,
        torch_device:   str        = "cuda",
    ):
        self._native_sr      = native_sr
        self._conf_thresh    = conf_thresh
        self._infer_interval = 1.0 / infer_hz
        self._torch_device   = torch_device
        self._id_to_label    = _build_id_to_label(class_map_path)

        # window lengths
        self._window_48k = self.WINDOW_FRAMES * self.HOP_SAMPLES        # 46 080
        self._window_44k = int(self._window_48k * native_sr / self.SLED_SR)  # ≈42 336

        # ── load model ───────────────────────────────────────────────────────
        print(f"[SLED] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=torch_device)

        use_hrtf_corr = ckpt.get("use_hrtf_corr", True)
        use_ild       = ckpt.get("use_ild",        True)
        use_ipd       = ckpt.get("use_ipd",        True)

        state = ckpt["model"]
        if "n_classes" in ckpt:
            n_classes = ckpt["n_classes"]
        elif "heads.class_head.weight" in state:
            n_classes = state["heads.class_head.weight"].shape[0]
        else:
            n_classes = 209

        self._model = SLEDv3(
            sofa_path     = os.path.abspath(sofa_path),
            d_model       = 256,
            n_classes     = n_classes,
            use_hrtf_corr = use_hrtf_corr,
            use_ild       = use_ild,
            use_ipd       = use_ipd,
        ).to(torch_device)
        self._model.load_state_dict(state)
        self._model.eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"[SLED] ready  n_classes={n_classes}  params={n_params:,}  "
              f"device={torch_device}")

        # ── shared state (inference thread → render thread) ──────────────────
        self._lock         = threading.Lock()
        self._latest_doa   = None   # np.ndarray [n_slots, 3]
        self._latest_conf  = None   # np.ndarray [n_slots]
        self._latest_cls   = None   # np.ndarray [n_slots]  int class ids
        self._audio_engine = None

        # ── background inference thread ──────────────────────────────────────
        self._stop   = threading.Event()
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="sled-infer"
        )
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────────

    def set_audio_engine(self, engine) -> None:
        """Attach the BinauralAudioEngine whose recording buffer we read."""
        self._audio_engine = engine

    def draw_on_frame(self, frame: np.ndarray, cam_w: int, cam_h: int) -> None:
        """
        Overlay DOA predictions onto *frame* (BGR HxWx3 uint8, modified in place).

        Parameters
        ----------
        frame : agentview OpenCV frame
        cam_w : frame pixel width
        cam_h : frame pixel height
        """
        with self._lock:
            if self._latest_doa is None:
                return
            doa  = self._latest_doa.copy()   # [S, 3]
            conf = self._latest_conf.copy()   # [S]
            cls  = self._latest_cls.copy()    # [S]

        # Focal length (square-pixel pinhole, fovy = 45°)
        f  = cam_h / (2.0 * np.tan(np.radians(self.FOVY_DEG / 2.0)))
        cx = cam_w / 2.0
        cy = cam_h / 2.0

        for s in range(doa.shape[0]):
            dx, dy, dz = doa[s]    # x=forward, y=right, z=up (SLED convention)
            c = float(conf[s])
            if c < self._conf_thresh:
                continue
            if dx < 0.05:          # source behind / tangent to camera plane
                continue

            u = int(round(cx + f * (dy / dx)))
            v = int(round(cy - f * (dz / dx)))   # minus: z up → v down

            if not (0 <= u < cam_w and 0 <= v < cam_h):
                continue

            color      = self._COLORS[s % len(self._COLORS)]
            cls_id     = int(cls[s])
            label      = self._id_to_label.get(cls_id, f"cls{cls_id}")
            # Truncate long class names (underscore → space for readability)
            label_disp = label.replace("_", " ")[:18]

            cv2.circle(frame, (u, v), 9, color, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (u, v), color,
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            # Class label one line above, confidence below
            cv2.putText(frame, label_disp,
                        (u + 12, v - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f"p={c:.2f}",
                        (u + 12, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                        cv2.LINE_AA)

        # Legend
        cv2.putText(frame, "SLED DOA",
                    (5, cam_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                    cv2.LINE_AA)

    def stop(self) -> None:
        """Stop the inference thread gracefully."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        print("[SLED] inference thread stopped.")

    # ── inference thread ──────────────────────────────────────────────────────

    def _inference_loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._run_one()
            except Exception as exc:
                # Transient errors (e.g. empty buffer at episode boundary)
                # are silently swallowed so the thread stays alive.
                pass
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, self._infer_interval - elapsed))

    def _run_one(self):
        engine = self._audio_engine
        if engine is None:
            return

        audio = engine.get_recorded_audio()   # [N, 2] float32, thread-safe
        if len(audio) < self._window_44k:
            return

        # Take last window, shape → [2, W]
        chunk = audio[-self._window_44k:, :].T.astype(np.float32)

        # Resample 44 100 → 48 000 Hz  (exact ratio 160/147)
        chunk_48k = resample_poly(chunk, 160, 147, axis=1).astype(np.float32)

        # Peak-normalise (mirrors stream_viz.py pre-processing)
        peak = float(np.abs(chunk_48k).max())
        if peak < 1e-6:
            return
        chunk_48k = chunk_48k / peak * 0.5

        # Trim/pad to exactly window_48k samples
        n = chunk_48k.shape[1]
        if n > self._window_48k:
            chunk_48k = chunk_48k[:, -self._window_48k:]
        elif n < self._window_48k:
            pad = self._window_48k - n
            chunk_48k = np.pad(chunk_48k, ((0, 0), (pad, 0)))

        tensor = (torch.from_numpy(chunk_48k)
                       .unsqueeze(0)
                       .to(self._torch_device))   # [1, 2, 46080]

        with torch.no_grad():
            result = self._model(tensor, gt=None)

        pred   = result["layer_preds"][-1]
        t_last = pred["class_logits"].shape[1] - 1

        doa  = pred["doa_vec"][0, t_last].cpu().numpy()                    # [S, 3]
        conf = torch.sigmoid(pred["confidence"][0, t_last]).cpu().numpy()  # [S]
        cls  = pred["class_logits"][0, t_last].argmax(-1).cpu().numpy()    # [S]

        with self._lock:
            self._latest_doa  = doa
            self._latest_conf = conf
            self._latest_cls  = cls
