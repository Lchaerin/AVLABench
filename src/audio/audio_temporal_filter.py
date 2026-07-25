"""Temporal filtering of noisy top-K audio snapshots fed to a VLA policy.

Why this exists
---------------
Both the SmolVLA and pi0/openpi pipelines drive the policy from a *single*
per-step audio snapshot (see `src/eval/eval_smolvla_audio.py`):

    audio_snap = _oracle_snapshot(...)   # oracle mode  → 1 clean GT + noise draw
    audio_snap = _live_snapshot(...)     # real SLED    → latest SLED prediction

In **oracle mode** that one-frame snapshot is fine: the geometry is exact and
only a small, controlled amount of Gaussian/flip noise is injected, so the
policy sees essentially the right answer every step. In the **real** regime the
per-frame SLED output is far noisier — the azimuth jitters by tens of degrees,
the class ID flips occasionally, confidence wobbles, and slots blink in and out
between frames. Conditioning on a single such frame hands the policy a moving
target.

This module aggregates a short sliding window of recent snapshots and emits a
single *stabilised* snapshot in the exact same `{class_id, azimuth_deg,
elevation_deg, confidence}` layout, so it drops into the eval loop right where
the raw snapshot is produced — before `_build_policy_batch` / the audio prompt
builder consumes it.

What it does that a naive average cannot
----------------------------------------
* **Circular azimuth statistics.** Azimuth wraps at ±180°, so a plain mean of
  e.g. {+179°, -179°} gives a nonsensical 0° (front) instead of ~180° (back).
  We use a confidence-weighted *circular* mean (mean of unit vectors), which is
  correct across the wrap and naturally down-weights low-confidence frames.
* **Outlier rejection.** A single wild SLED azimuth (common during a bad
  window) is rejected before it can drag the estimate, via a circular-distance
  gate around the running mean.
* **Robust class voting.** Majority vote over the window with the same
  "prefer a real class over silence dropouts" rule as
  `oracle_sled.TopKClassSmoother`, generalised so it also stabilises azimuth /
  elevation / confidence rather than class alone.
* **Presence hysteresis (debounce).** A slot only turns ON after it has been
  confidently present in enough of the recent frames, and only turns OFF after
  it has been absent for enough frames. This suppresses single-frame phantom
  detections and rides through brief SLED dropouts instead of flickering the
  cue to silence.
* **Optional cross-frame tracking.** Real SLED does not guarantee a stable slot
  order between frames (slot 0 this frame may be slot 1 the next). The
  ``association="track"`` mode runs a tiny nearest-neighbour tracker that
  associates detections across frames by class + angular proximity, so the
  smoothing follows a *physical source* rather than a slot index. The default
  ``association="positional"`` mode assumes stable slot order (correct for the
  oracle and any static single-source scene) and is cheaper.

Conventions
-----------
Azimuth/elevation are in **degrees**, SLED sign convention (az > 0 ⇒ source on
the listener's RIGHT), matching `oracle_sled.gt_az_el` and `_live_snapshot`.
Silence is encoded as ``class_id = -1`` with ``confidence = 0``.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

# Snapshot field names — identical to the dicts produced by
# oracle_sled.build_oracle_topk / eval_smolvla_audio._live_snapshot.
_FIELDS = ("class_id", "azimuth_deg", "elevation_deg", "confidence")


# --------------------------------------------------------------------------- #
# Small angular helpers (degrees, circular at ±180)
# --------------------------------------------------------------------------- #
def _wrap180(deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    return (float(deg) + 180.0) % 360.0 - 180.0


def _ang_dist(a: float, b: float) -> float:
    """Absolute smallest angular distance between two bearings, in degrees."""
    return abs(_wrap180(a - b))


def _circular_mean(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean of bearings in degrees.

    Returns the angle of the (weighted) resultant unit vector, which is the
    only mean that behaves correctly across the ±180 wrap. Falls back to a
    plain weighted index-0 value if all weights are zero.
    """
    w = np.asarray(weights, dtype=np.float64)
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    if w.sum() <= 0:
        w = np.ones_like(w)
    s = float(np.sum(w * np.sin(a)))
    c = float(np.sum(w * np.cos(a)))
    if s == 0.0 and c == 0.0:
        return float(angles_deg[0])
    return float(np.degrees(np.arctan2(s, c)))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AudioFilterConfig:
    """Tunables for :class:`AudioTemporalFilter`.

    The defaults target ~10 Hz VLA stepping with a real SLED front-end; for the
    oracle pipeline a shorter ``window`` (3-5) is usually enough since the only
    noise is the injected Gaussian/flip.
    """
    top_k: int = 3
    window: int = 7                 # number of recent frames kept per slot/track

    # ── association across frames ─────────────────────────────────────────
    #   "positional" : slot index is stable across frames (oracle / static
    #                  single-source). Smooth each slot independently.
    #   "track"      : slot order may permute (real multi-source SLED). Run a
    #                  nearest-neighbour tracker so smoothing follows a source.
    association: str = "positional"
    assoc_max_az_deg: float = 30.0  # gate: a detection joins a track only if
                                    # within this bearing of the track estimate
    assoc_class_bonus_deg: float = 60.0  # effective gate widening when the
                                         # detection's class matches the track

    # ── class voting ──────────────────────────────────────────────────────
    silence_id: int = -1
    prefer_real_class: bool = True  # drop silence votes when a real class is
                                    # present in the window (ride out dropouts)
    min_class_votes: int = 1        # a real class needs ≥ this many votes to win

    # ── azimuth / elevation smoothing ────────────────────────────────────
    conf_weighted: bool = True      # weight angle samples by their confidence
    az_outlier_deg: float = 45.0    # reject az samples this far from the running
                                    # circular mean (<=0 disables rejection)
    el_estimator: str = "median"    # "median" | "mean"

    # ── confidence smoothing ──────────────────────────────────────────────
    conf_estimator: str = "median"  # "median" | "mean" | "ema"
    conf_ema_alpha: float = 0.4     # weight of the newest frame for "ema"

    # ── presence hysteresis (debounce) ───────────────────────────────────
    present_conf_min: float = 0.10  # a frame counts as "present" above this conf
    on_fraction: float = 0.5        # turn a slot ON when this fraction of the
                                    # window is present
    off_fraction: float = 0.2       # turn it OFF when present fraction drops
                                    # below this (otherwise hold the last cue)

    def __post_init__(self) -> None:
        if self.association not in ("positional", "track"):
            raise ValueError(f"association must be positional|track, got {self.association!r}")
        if self.el_estimator not in ("median", "mean"):
            raise ValueError(f"el_estimator must be median|mean, got {self.el_estimator!r}")
        if self.conf_estimator not in ("median", "mean", "ema"):
            raise ValueError(f"conf_estimator must be median|mean|ema, got {self.conf_estimator!r}")
        self.top_k = int(max(1, self.top_k))
        self.window = int(max(1, self.window))


# --------------------------------------------------------------------------- #
# A single per-slot / per-track observation history
# --------------------------------------------------------------------------- #
class _History:
    """Sliding window of observations for one slot (positional) or one source
    (track). Stores the raw per-frame (cid, az, el, conf) and computes the
    smoothed estimates on demand."""

    __slots__ = ("cfg", "buf", "_on", "last_seen", "_ema_conf")

    def __init__(self, cfg: AudioFilterConfig):
        self.cfg = cfg
        self.buf: Deque[tuple] = deque(maxlen=cfg.window)
        self._on = False            # presence-hysteresis latch
        self.last_seen = -1         # frame index of last present observation
        self._ema_conf = 0.0

    # -- ingestion ---------------------------------------------------------
    def push(self, cid: int, az: float, el: float, conf: float, frame: int) -> None:
        present = (int(cid) != self.cfg.silence_id) and (float(conf) > self.cfg.present_conf_min)
        self.buf.append((int(cid), float(az), float(el), float(conf), bool(present)))
        if present:
            self.last_seen = frame
        a = self.cfg.conf_ema_alpha
        self._ema_conf = (1 - a) * self._ema_conf + a * float(conf)

    def push_absent(self, frame: int) -> None:
        """Record a frame in which this slot/track produced no detection."""
        self.push(self.cfg.silence_id, 0.0, 0.0, 0.0, frame)

    # -- queries -----------------------------------------------------------
    def _present_obs(self) -> List[tuple]:
        return [o for o in self.buf if o[4]]

    def present_fraction(self) -> float:
        if not self.buf:
            return 0.0
        return sum(1 for o in self.buf if o[4]) / len(self.buf)

    def update_presence(self) -> bool:
        """Advance the ON/OFF hysteresis latch and return the current state."""
        frac = self.present_fraction()
        if self._on:
            if frac < self.cfg.off_fraction:
                self._on = False
        else:
            if frac >= self.cfg.on_fraction:
                self._on = True
        return self._on

    def voted_class(self) -> int:
        cfg = self.cfg
        cids = [o[0] for o in self.buf]
        if not cids:
            return cfg.silence_id
        if cfg.prefer_real_class:
            real = [c for c in cids if c != cfg.silence_id]
            if real:
                cids = real
        cid, votes = Counter(cids).most_common(1)[0]
        if cid != cfg.silence_id and votes < cfg.min_class_votes:
            return cfg.silence_id
        return int(cid)

    def smoothed(self) -> tuple:
        """Return the stabilised (cid, az, el, conf) for this history.

        Angle/confidence are computed over the *present* frames whose class
        matches the voted class, so a stray off-class frame cannot contaminate
        the direction estimate.
        """
        cfg = self.cfg
        cid = self.voted_class()
        if cid == cfg.silence_id:
            return cfg.silence_id, 0.0, 0.0, 0.0

        obs = [o for o in self._present_obs() if o[0] == cid]
        if not obs:                              # voted real class but only
            obs = self._present_obs()            # off-class present frames left
        if not obs:
            return cid, 0.0, 0.0, 0.0

        az = np.array([o[1] for o in obs], dtype=np.float64)
        el = np.array([o[2] for o in obs], dtype=np.float64)
        conf = np.array([o[3] for o in obs], dtype=np.float64)
        w = conf.copy() if cfg.conf_weighted else np.ones_like(conf)

        # Azimuth: circular mean with a one-pass outlier rejection.
        az_est = _circular_mean(az, w)
        if cfg.az_outlier_deg > 0 and az.size > 2:
            keep = np.array([_ang_dist(a, az_est) <= cfg.az_outlier_deg for a in az])
            if keep.any() and not keep.all():
                az_est = _circular_mean(az[keep], w[keep])
                el = el[keep]
                conf_in = conf[keep]
            else:
                conf_in = conf
        else:
            conf_in = conf

        # Elevation: robust central tendency (no wrap concerns in [-90, 90]).
        el_est = float(np.median(el)) if cfg.el_estimator == "median" else float(np.average(el))

        # Confidence: median / mean over present frames, or the running EMA.
        if cfg.conf_estimator == "ema":
            conf_est = float(self._ema_conf)
        elif cfg.conf_estimator == "mean":
            conf_est = float(np.mean(conf_in))
        else:
            conf_est = float(np.median(conf_in))

        return cid, float(_wrap180(az_est)), el_est, conf_est


# --------------------------------------------------------------------------- #
# Public filter
# --------------------------------------------------------------------------- #
class AudioTemporalFilter:
    """Stateful temporal filter over per-step top-K audio snapshots.

    Usage (mirrors :class:`oracle_sled.TopKClassSmoother`)::

        filt = AudioTemporalFilter(AudioFilterConfig(top_k=3, window=7))
        filt.reset()                       # once per episode
        for step in episode:
            raw  = _live_snapshot(...)      # or _oracle_snapshot(...)
            snap = filt.update(raw)         # stabilised, same dict layout
            batch = _build_policy_batch(env, snap, ...)

    ``update`` is pure w.r.t. its argument (it copies), and always returns a
    dict with numpy arrays of length ``top_k`` for every field, so it is a
    drop-in replacement for the raw snapshot.
    """

    def __init__(self, config: Optional[AudioFilterConfig] = None, **kwargs):
        if config is None:
            config = AudioFilterConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a config or keyword args, not both")
        self.cfg = config
        self._frame = 0
        # positional: one fixed history per slot.
        self._slots: List[_History] = [_History(config) for _ in range(config.top_k)]
        # track mode: a growing list of source tracks.
        self._tracks: List[_History] = []

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all history. Call once at the start of each episode."""
        self._frame = 0
        self._slots = [_History(self.cfg) for _ in range(self.cfg.top_k)]
        self._tracks = []

    # ------------------------------------------------------------------ #
    def update(self, snap: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ingest one raw snapshot and return the stabilised snapshot."""
        cid, az, el, conf = self._unpack(snap)
        if self.cfg.association == "positional":
            out = self._update_positional(cid, az, el, conf)
        else:
            out = self._update_track(cid, az, el, conf)
        self._frame += 1
        return out

    # also accept the TopKClassSmoother call site name as an alias.
    smooth_snapshot = update

    # ------------------------------------------------------------------ #
    def _unpack(self, snap):
        k = self.cfg.top_k
        def _vec(key, dtype):
            v = np.asarray(snap[key]).reshape(-1).astype(dtype)
            if v.size < k:                       # pad short snapshots with silence
                pad = np.zeros(k - v.size, dtype=dtype)
                if key == "class_id":
                    pad[:] = self.cfg.silence_id
                v = np.concatenate([v, pad])
            return v[:k]
        return (_vec("class_id", np.int64),
                _vec("azimuth_deg", np.float64),
                _vec("elevation_deg", np.float64),
                _vec("confidence", np.float64))

    def _empty_out(self):
        k = self.cfg.top_k
        return {
            "class_id":      np.full(k, self.cfg.silence_id, dtype=np.int32),
            "azimuth_deg":   np.zeros(k, dtype=np.float32),
            "elevation_deg": np.zeros(k, dtype=np.float32),
            "confidence":    np.zeros(k, dtype=np.float32),
        }

    # ----- positional: stable slot order ------------------------------ #
    def _update_positional(self, cid, az, el, conf):
        out = self._empty_out()
        for s in range(self.cfg.top_k):
            hist = self._slots[s]
            hist.push(int(cid[s]), float(az[s]), float(el[s]), float(conf[s]), self._frame)
            on = hist.update_presence()
            if not on:
                continue
            c, a, e, cf = hist.smoothed()
            if c == self.cfg.silence_id:
                continue
            out["class_id"][s] = c
            out["azimuth_deg"][s] = a
            out["elevation_deg"][s] = e
            out["confidence"][s] = cf
        return out

    # ----- track: associate detections to physical sources ------------ #
    def _update_track(self, cid, az, el, conf):
        cfg = self.cfg
        # Detections present this frame.
        dets = [(int(cid[i]), float(az[i]), float(el[i]), float(conf[i]))
                for i in range(cfg.top_k)
                if int(cid[i]) != cfg.silence_id and float(conf[i]) > cfg.present_conf_min]

        # Greedy nearest-neighbour association by ascending gated cost.
        track_est = [t.smoothed() for t in self._tracks]
        pairs = []
        for di, d in enumerate(dets):
            for ti, est in enumerate(track_est):
                gate = cfg.assoc_max_az_deg
                if est[0] == d[0]:               # class match → wider gate
                    gate = max(gate, cfg.assoc_class_bonus_deg)
                dist = _ang_dist(d[1], est[1])
                if dist <= gate:
                    pairs.append((dist, di, ti))
        pairs.sort(key=lambda p: p[0])

        det_used = [False] * len(dets)
        trk_used = [False] * len(self._tracks)
        matched: Dict[int, int] = {}             # track idx -> det idx
        for _, di, ti in pairs:
            if det_used[di] or trk_used[ti]:
                continue
            det_used[di], trk_used[ti] = True, True
            matched[ti] = di

        # Update matched tracks; mark unmatched existing tracks absent.
        for ti, t in enumerate(self._tracks):
            if ti in matched:
                d = dets[matched[ti]]
                t.push(d[0], d[1], d[2], d[3], self._frame)
            else:
                t.push_absent(self._frame)
            t.update_presence()

        # Spawn new tracks for unmatched detections.
        for di, d in enumerate(dets):
            if det_used[di]:
                continue
            t = _History(cfg)
            t.push(d[0], d[1], d[2], d[3], self._frame)
            t.update_presence()
            self._tracks.append(t)

        # Retire tracks unseen for a whole window.
        self._tracks = [t for t in self._tracks
                        if (self._frame - t.last_seen) < cfg.window]

        # Emit the top_k confirmed (ON) tracks, strongest confidence first.
        out = self._empty_out()
        confirmed = []
        for t in self._tracks:
            if not t._on:
                continue
            c, a, e, cf = t.smoothed()
            if c != cfg.silence_id:
                confirmed.append((cf, c, a, e))
        confirmed.sort(key=lambda x: -x[0])
        for s, (cf, c, a, e) in enumerate(confirmed[:cfg.top_k]):
            out["class_id"][s] = c
            out["azimuth_deg"][s] = a
            out["elevation_deg"][s] = e
            out["confidence"][s] = cf
        return out


# --------------------------------------------------------------------------- #
# Convenience factory mirroring the eval CLI flags
# --------------------------------------------------------------------------- #
def build_filter_from_args(args) -> Optional[AudioTemporalFilter]:
    """Construct a filter from an argparse Namespace, or None if disabled.

    Reads these (all optional) attributes, defaulting sensibly when absent so
    it can be wired into the existing eval scripts without touching unrelated
    flags:
        --audio-filter-window     (int, <=1 disables the filter)
        --audio-filter-assoc      (positional|track)
        --top-k                   (slot count)
    """
    window = int(getattr(args, "audio_filter_window", 0) or 0)
    if window <= 1:
        return None
    cfg = AudioFilterConfig(
        top_k=int(getattr(args, "top_k", 3)),
        window=window,
        association=str(getattr(args, "audio_filter_assoc", "positional")),
    )
    return AudioTemporalFilter(cfg)
