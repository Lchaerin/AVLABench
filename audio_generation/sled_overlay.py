"""
SLEDOverlay — integrates SLEDv3/v4/v5 sound-source localisation with AVLABench.

Two usage modes
---------------
Real-time (realtime=True, default):
    A background thread polls the attached BinauralAudioEngine recording buffer
    at *infer_hz* and updates self._latest_pred.  Use set_audio_engine() then
    get_latest_prediction() / draw_on_frame().

Post-hoc (realtime=False):
    No background thread.  After the episode, call run_post_hoc(audio_arr,
    n_frames) to get per-frame predictions from the saved WAV array.
    This is the recommended mode for trajectory_generation.py because the
    MuJoCo simulation runs much faster than real-time, so the audio buffer
    rarely accumulates enough data during the live run.

Supported checkpoint formats (auto-detected)
--------------------------------------------
v3  : sled/model/sled.py → SLEDv3
        ckpt keys: "model", "use_hrtf_corr", "use_ild", "use_ipd"  (no "config")
v4  : sled_v4/sled.py → SLEDv4
        ckpt keys: "model", "config"  (SLEDConfig dataclass dict)
v5  : sled_v5/models/biseld_net.py → BiSELDNet
        ckpt keys: "model_state", "config"  (YAML dict, 24 kHz)

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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve crossCorr root (two levels up from audio_generation/)
# ---------------------------------------------------------------------------
_HERE              = os.path.dirname(os.path.abspath(__file__))
_CROSSCORR_ROOT    = os.path.normpath(os.path.join(_HERE, "../../crossCorr"))
_DEFAULT_CKPT      = os.path.join(_CROSSCORR_ROOT,
                                   "checkpoints_ver8_50db", "sled_best.pt")
_DEFAULT_SOFA      = os.path.join(_CROSSCORR_ROOT, "hrtf", "custom_mrs.sofa")
_DEFAULT_CLASS_MAP = os.path.join(_CROSSCORR_ROOT, "data", "meta", "class_map.json")

if _CROSSCORR_ROOT not in sys.path:
    sys.path.insert(0, _CROSSCORR_ROOT)


def _build_id_to_label(class_map_path: str | None) -> dict[int, str]:
    """Build {class_id: label} from a class_map.json (key = "ClassName/file.wav")."""
    if not class_map_path or not os.path.exists(class_map_path):
        return {}
    with open(class_map_path) as f:
        cm = json.load(f)
    id_to_label: dict[int, str] = {}
    for key, cid in cm.items():
        label = key.split("/")[0]
        if cid not in id_to_label:
            id_to_label[cid] = label
    return id_to_label


def _detect_version(ckpt: dict) -> str:
    """Auto-detect checkpoint version from saved keys."""
    if "model_state" in ckpt:
        return "v5"
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        # v4 config is a dataclass-derived dict with "d_model" key
        # v5 config has "model" / "audio" sub-dicts → already handled above
        return "v4"
    return "v3"


# ---------------------------------------------------------------------------
# Per-version model loaders
# ---------------------------------------------------------------------------

def _load_model_v3(ckpt: dict, sofa_path: str, device: str):
    """Load SLEDv3 from a checkpoint that has no 'config' sub-dict."""
    from sled.model.sled import SLEDv3

    use_hrtf_corr = ckpt.get("use_hrtf_corr", True)
    use_ild       = ckpt.get("use_ild",        True)
    use_ipd       = ckpt.get("use_ipd",        True)

    model = SLEDv3(
        sofa_path         = os.path.abspath(sofa_path),
        d_model           = 256,
        n_slots           = 12,
        n_classes         = 209,
        max_sources       = 3,
        n_decoder_layers  = 4,
        n_conformer_layers= 6,
        use_hrtf_corr     = use_hrtf_corr,
        use_ild           = use_ild,
        use_ipd           = use_ipd,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_model_v4(ckpt: dict, sofa_path: str, device: str):
    """Load SLEDv4 from a checkpoint that has a 'config' sub-dict."""
    from sled_v4.config import SLEDConfig
    from sled_v4.sled   import SLEDv4

    raw_cfg = ckpt["config"]
    cfg = SLEDConfig(
        d_model              = raw_cfg.get("d_model",              256),
        n_classes            = raw_cfg.get("n_classes",            209),
        n_slots              = raw_cfg.get("n_slots",              12),
        max_sources          = raw_cfg.get("max_sources",          3),
        sr                   = raw_cfg.get("sr",                   48_000),
        n_fft                = raw_cfg.get("n_fft",                2048),
        hop_length           = raw_cfg.get("hop_length",           960),
        n_mels               = raw_cfg.get("n_mels",               64),
        use_hrtf_corr        = raw_cfg.get("use_hrtf_corr",        True),
        use_ild              = raw_cfg.get("use_ild",              True),
        use_ipd              = raw_cfg.get("use_ipd",              True),
        stem_channels        = tuple(raw_cfg.get("stem_channels",  (64, 128, 256))),
        n_bifpn              = raw_cfg.get("n_bifpn",              2),
        n_conformer          = raw_cfg.get("n_conformer",          6),
        conformer_ffn_dim    = raw_cfg.get("conformer_ffn_dim",    512),
        conformer_conv_kernel= raw_cfg.get("conformer_conv_kernel",31),
        conformer_dropout    = raw_cfg.get("conformer_dropout",    0.1),
        drop_path_rate       = raw_cfg.get("drop_path_rate",       0.0),
        n_az                 = raw_cfg.get("n_az",                 36),
        n_el                 = raw_cfg.get("n_el",                 2),
        n_decoder_layers     = raw_cfg.get("n_decoder_layers",     4),
        decoder_ffn_dim      = raw_cfg.get("decoder_ffn_dim",      1024),
        n_dn_groups          = raw_cfg.get("n_dn_groups",          3),
        is_teacher           = raw_cfg.get("is_teacher",           False),
    )
    model = SLEDv4(
        cfg       = cfg,
        sofa_path = os.path.abspath(sofa_path),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg.n_classes


def _load_model_v5(ckpt: dict, device: str):
    """Load BiSELDNet (v5) from a checkpoint that has 'model_state'."""
    _v5_dir = os.path.join(_CROSSCORR_ROOT, "sled_v5")
    if _v5_dir not in sys.path:
        sys.path.insert(0, _v5_dir)

    from models.biseld_net import build_model  # noqa: E402

    cfg   = ckpt.get("config", {})
    model = build_model(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    # mel_transform buffers differ from torchaudio → recomputed at init; other
    # keys must all be accounted for.
    _mel_keys = {"btff.mel_transform.spectrogram.window",
                 "btff.mel_transform.mel_scale.fb",
                 "btff.mel_transform.fb"}
    bad_missing    = [k for k in missing    if k not in _mel_keys]
    bad_unexpected = [k for k in unexpected if k not in _mel_keys]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"[SLED v5] Unexpected state-dict mismatch — "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    model.eval()

    activity_thresh = (
        cfg.get("inference", {}).get("activity_threshold")
        or cfg.get("training", {}).get("activity_threshold", 0.5)
    )
    n_classes = cfg.get("model", {}).get("n_classes", 14)
    max_sources = cfg.get("model", {}).get("n_tracks", 3)
    return model, activity_thresh, n_classes, max_sources


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SLEDOverlay:
    """
    Loads a SLED checkpoint (v3, v4, or v5); supports real-time and post-hoc
    inference.

    Parameters
    ----------
    ckpt_path       : path to sled_best.pt / biseld_best.pt / teacher_best.pt
    sofa_path       : HRTF SOFA file (used by v3 / v4 preprocessor)
    class_map_path  : optional path to class_map.json for human-readable labels
    native_sr       : audio engine sample rate in Hz  (default 44 100)
    conf_thresh     : minimum slot confidence to draw a marker
    infer_hz        : target inference rate for real-time mode (Hz)
    torch_device    : 'cuda' or 'cpu'
    fovy_deg        : vertical field of view of the target camera in degrees
    realtime        : if False, skip the background inference thread
    """

    # v3 / v4 constants
    SLED_SR_V3V4   = 48_000
    WINDOW_FRAMES  = 48            # 48 × 960 = 46 080 samples @ 48 kHz (~960 ms)
    HOP_SAMPLES    = 960

    # v5 constants
    SLED_SR_V5     = 24_000
    HOP_SAMPLES_V5 = 240           # 48 × 240 = 11 520 samples @ 24 kHz (~480 ms)

    # BGR colours for slot 0, 1, 2
    _COLORS = [
        (0x3c, 0x4c, 0xe7),   # blue
        (0x71, 0xcc, 0x2e),   # green
        (0xdb, 0x98, 0x34),   # orange
    ]

    def __init__(
        self,
        ckpt_path:      str   = _DEFAULT_CKPT,
        sofa_path:      str   = _DEFAULT_SOFA,
        class_map_path: str | None = _DEFAULT_CLASS_MAP,
        native_sr:      int   = 44_100,
        conf_thresh:    float = 0.30,
        infer_hz:       float = 5.0,
        torch_device:   str   = "cuda",
        fovy_deg:       float = 45.0,
        realtime:       bool  = False,
    ):
        self._native_sr      = native_sr
        self._conf_thresh    = conf_thresh
        self._infer_interval = 1.0 / infer_hz
        self._torch_device   = torch_device
        self._fovy_deg       = fovy_deg
        self._id_to_label    = _build_id_to_label(class_map_path)

        # ── load checkpoint & detect version ────────────────────────────────
        print(f"[SLED] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=torch_device, weights_only=False)

        self._model_version = _detect_version(ckpt)
        print(f"[SLED] detected version : {self._model_version}")

        # ── version-specific init ────────────────────────────────────────────
        if self._model_version == "v5":
            self._sled_sr       = self.SLED_SR_V5
            window_sled         = self.WINDOW_FRAMES * self.HOP_SAMPLES_V5   # 11520
            self._window_sled   = window_sled
            self._window_native = int(window_sled * native_sr / self.SLED_SR_V5)

            (self._model,
             self._activity_thresh,
             n_classes,
             self._max_sources) = _load_model_v5(ckpt, torch_device)

        else:
            # v3 and v4 share the same SR and output format
            self._sled_sr       = self.SLED_SR_V3V4
            window_sled         = self.WINDOW_FRAMES * self.HOP_SAMPLES       # 46080
            self._window_sled   = window_sled
            self._window_native = int(window_sled * native_sr / self.SLED_SR_V3V4)

            if self._model_version == "v4":
                self._model, n_classes = _load_model_v4(ckpt, sofa_path, torch_device)
            else:  # v3
                self._model = _load_model_v3(ckpt, sofa_path, torch_device)
                n_classes   = getattr(self._model, "n_classes", 209)

        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"[SLED] ready  version={self._model_version}  "
              f"n_classes={n_classes}  params={n_params:,}  "
              f"device={torch_device}  realtime={realtime}")

        # ── shared state (inference thread → render thread) ─────────────────
        self._lock         = threading.Lock()
        self._latest_pred  = None
        self._audio_engine = None

        # ── optional background inference thread ─────────────────────────────
        self._stop   = threading.Event()
        self._thread = None
        if realtime:
            self._thread = threading.Thread(
                target=self._inference_loop, daemon=True, name="sled-infer"
            )
            self._thread.start()

    # ── public API ─────────────────────────────────────────────────────────

    def set_audio_engine(self, engine) -> None:
        """Attach the BinauralAudioEngine whose recording buffer we read (real-time mode)."""
        self._audio_engine = engine

    def get_latest_prediction(self):
        """
        Return a thread-safe snapshot: (doa [S,3], conf [S], cls [S]) or None.
        """
        with self._lock:
            if self._latest_pred is None:
                return None
            doa, conf, cls = self._latest_pred
            return doa.copy(), conf.copy(), cls.copy()

    def run_post_hoc(
        self,
        audio_arr: np.ndarray,
        n_frames:  int,
        video_fps: float = 10.0,
    ) -> list:
        """
        Run SLED inference frame-by-frame on a saved audio array.

        For each video frame i the corresponding audio window ends at
        t = i / video_fps seconds; the window length is ~960 ms (v3/v4)
        or ~480 ms (v5).

        Parameters
        ----------
        audio_arr : [N, 2] float32 at self._native_sr Hz
        n_frames  : total number of video frames
        video_fps : video frame rate (default 10)

        Returns
        -------
        list of length n_frames, each element is
        (doa [S,3], conf [S], cls [S]) or None when not enough audio yet.
        """
        predictions = []
        print(f"[SLED] post-hoc inference ({self._model_version}): "
              f"{n_frames} frames @ {video_fps:.2f} fps …")
        for frame_idx in tqdm(range(n_frames), desc="[SLED] post-hoc"):
            pred = self._infer_chunk_at(audio_arr, frame_idx, video_fps)
            predictions.append(pred)
            if pred is not None and not hasattr(self, "_debug_printed"):
                doa, conf, cls = pred
                print(f"[SLED] first prediction at frame {frame_idx}: "
                      f"conf={conf.tolist()}  doa={doa.tolist()}")
                self._debug_printed = True

        import math as _math
        _f = 1.0 / (2.0 * _math.tan(_math.radians(self._fovy_deg / 2.0)))
        n_valid   = sum(p is not None for p in predictions)
        n_markers = 0
        for p in predictions:
            if p is None:
                continue
            doa, conf, cls = p
            for s in range(doa.shape[0]):
                dx, dy, dz = doa[s]
                if float(conf[s]) < self._conf_thresh or dx < 0.05:
                    continue
                u_n = 0.5 + _f * (dy / dx)
                v_n = 0.5 - _f * (dz / dx)
                if 0.0 <= u_n <= 1.0 and 0.0 <= v_n <= 1.0:
                    n_markers += 1
                    break
        print(f"[SLED] {n_valid}/{n_frames} frames with predictions, "
              f"{n_markers} frames with visible markers "
              f"(conf≥{self._conf_thresh}, dx≥0.05, in-bounds)")
        return predictions

    def draw_on_frame(self, frame: np.ndarray, cam_w: int, cam_h: int) -> None:
        """Overlay the latest real-time prediction onto *frame* (BGR, in place)."""
        pred = self.get_latest_prediction()
        if pred is None:
            return
        doa, conf, cls = pred
        self._draw(frame, doa, conf, cls, cam_w, cam_h)

    def draw_with_pred(
        self,
        frame:  np.ndarray,
        pred,
        cam_w:  int,
        cam_h:  int,
    ) -> None:
        """Overlay a specific prediction snapshot onto *frame* (BGR, in place)."""
        if pred is None:
            return
        doa, conf, cls = pred
        self._draw(frame, doa, conf, cls, cam_w, cam_h)

    def stop(self) -> None:
        """Stop the real-time inference thread gracefully (no-op in post-hoc mode)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            print("[SLED] inference thread stopped.")

    # ── core inference (shared by real-time and post-hoc) ──────────────────

    def _infer_chunk_at(
        self,
        audio_arr: np.ndarray,   # [N, 2] float32 @ native_sr
        frame_idx: int,
        video_fps: float,
    ):
        """Return (doa, conf, cls) for the audio window ending at frame_idx, or None."""
        t_samples = int(frame_idx / video_fps * self._native_sr)
        end   = min(len(audio_arr), t_samples + 1)
        start = max(0, end - self._window_native)

        if end <= 0 or (end - start) < self._window_native // 8:
            return None

        chunk = audio_arr[start:end, :].T.astype(np.float32)   # [2, L]

        if chunk.shape[1] < self._window_native:
            chunk = np.pad(chunk, ((0, 0), (self._window_native - chunk.shape[1], 0)))

        return self._infer_raw_chunk(chunk)

    # ── rendering ──────────────────────────────────────────────────────────

    def _draw(
        self,
        frame: np.ndarray,
        doa:   np.ndarray,  # [S, 3]
        conf:  np.ndarray,  # [S]
        cls:   np.ndarray,  # [S]
        cam_w: int,
        cam_h: int,
    ) -> None:
        f  = cam_h / (2.0 * np.tan(np.radians(self._fovy_deg / 2.0)))
        cx = cam_w / 2.0
        cy = cam_h / 2.0

        for s in range(doa.shape[0]):
            dx, dy, dz = doa[s]   # x=forward, y=right, z=up
            c = float(conf[s])
            if c < self._conf_thresh:
                continue
            if dx < 0.05:
                continue

            u = int(round(cx + f * (dy / dx)))
            v = int(round(cy - f * (dz / dx)))

            if not (0 <= u < cam_w and 0 <= v < cam_h):
                continue

            color      = self._COLORS[s % len(self._COLORS)]
            cls_id     = int(cls[s])
            label      = self._id_to_label.get(cls_id, f"cls{cls_id}")
            label_disp = label.replace("_", " ")[:18]

            cv2.circle(frame, (u, v), 9, color, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (u, v), color,
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(frame, label_disp,
                        (u + 12, v - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"p={c:.2f}",
                        (u + 12, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"SLED {self._model_version} DOA",
                    (5, cam_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # ── real-time inference thread ─────────────────────────────────────────

    def _inference_loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._run_one_realtime()
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, self._infer_interval - elapsed))

    def _run_one_realtime(self):
        engine = self._audio_engine
        if engine is None:
            return
        audio = engine.get_recorded_audio()
        if len(audio) < self._window_native:
            return
        chunk = audio[-self._window_native:, :].T.astype(np.float32)
        pred = self._infer_raw_chunk(chunk)
        if pred is not None:
            with self._lock:
                self._latest_pred = pred

    def _infer_raw_chunk(self, chunk: np.ndarray):
        """
        Run one inference on a [2, window_native] float32 chunk at native_sr.
        Dispatches to the appropriate version-specific method.
        Returns (doa [S,3], conf [S], cls [S]) or None.
        """
        if self._model_version == "v5":
            return self._infer_raw_chunk_v5(chunk)
        else:
            return self._infer_raw_chunk_v3v4(chunk)

    def _infer_raw_chunk_v3v4(self, chunk: np.ndarray):
        """Inference for SLEDv3 / SLEDv4: resample 44100 → 48000, slot-based output."""
        chunk_48k = resample_poly(chunk, 160, 147, axis=1).astype(np.float32)

        n = chunk_48k.shape[1]
        if n > self._window_sled:
            chunk_48k = chunk_48k[:, -self._window_sled:]
        elif n < self._window_sled:
            chunk_48k = np.pad(chunk_48k, ((0, 0), (self._window_sled - n, 0)))

        peak = float(np.abs(chunk_48k).max())
        if peak < 1e-6:
            return None
        chunk_48k = chunk_48k / peak * 0.5

        tensor = (torch.from_numpy(chunk_48k)
                       .unsqueeze(0)
                       .to(self._torch_device))
        with torch.no_grad():
            result = self._model(tensor, gt=None)

        pred   = result["layer_preds"][-1]
        t_last = pred["class_logits"].shape[1] - 1
        doa  = pred["doa_vec"][0, t_last].cpu().numpy()
        conf = torch.sigmoid(pred["confidence"][0, t_last]).cpu().numpy()
        cls  = pred["class_logits"][0, t_last].argmax(-1).cpu().numpy()
        return doa, conf, cls

    def _infer_raw_chunk_v5(self, chunk: np.ndarray):
        """Inference for BiSELDNet (v5): resample 44100 → 24000, Multi-ACCDOA output."""
        # Resample 44100 → 24000  (ratio = 80/147)
        chunk_24k = resample_poly(chunk, 80, 147, axis=1).astype(np.float32)

        n = chunk_24k.shape[1]
        if n > self._window_sled:
            chunk_24k = chunk_24k[:, -self._window_sled:]
        elif n < self._window_sled:
            chunk_24k = np.pad(chunk_24k, ((0, 0), (self._window_sled - n, 0)))

        peak = float(np.abs(chunk_24k).max())
        if peak < 1e-6:
            return None
        chunk_24k = chunk_24k / peak * 0.5

        tensor = (torch.from_numpy(chunk_24k)
                       .unsqueeze(0)
                       .to(self._torch_device))
        with torch.no_grad():
            pred, _ = self._model(tensor)   # (1, T_d, N, C, 3)

        # Decode last time frame: (N, C, 3)
        last_frame = pred[0, -1].cpu()
        N, C, _ = last_frame.shape

        active = []
        for n_track in range(N):
            for c in range(C):
                vec      = last_frame[n_track, c]
                activity = vec.norm().item()
                if activity > self._activity_thresh:
                    unit = (vec / (activity + 1e-8)).numpy()
                    active.append((activity, c, unit))

        if not active:
            return None

        # Sort by activity descending, keep top max_sources
        active.sort(key=lambda x: -x[0])
        active = active[:self._max_sources]

        doa  = np.array([u     for _, _, u in active], dtype=np.float32)
        conf = np.array([a     for a, _, _ in active], dtype=np.float32)
        cls  = np.array([c     for _, c, _ in active], dtype=np.int64)

        # Clip conf to [0, 1] for display (activity can exceed 1)
        conf = np.clip(conf, 0.0, 1.0)

        return doa, conf, cls
