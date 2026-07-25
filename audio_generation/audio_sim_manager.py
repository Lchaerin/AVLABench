"""
AudioSimManager – integrates BinauralAudioEngine with AVLABench simulation.

Design
------
* Reads a JSON config that maps task names → list of {object_name, sound_file, gain}.
* Randomly picks one TAU-SRIR .mat file and its matching TAU-SNoise directory
  (matched by the 2-digit numeric prefix, e.g. "04_pc226").
* Uses the single provided SOFA file for HRTF (audio_generation/hrtf/p0001.sofa).
* Monkey-patches env.step so that listener-relative azimuth/elevation are
  recomputed from the main camera every simulation step.
* After the episode, stop() records the audio and saves it as a WAV file.

Usage
-----
    mgr = AudioSimManager(
        config_path="audio_generation/scene_audio_config.json",
        task_name="select_toy",
    )
    mgr.attach_to_env(env)   # patches env.step
    mgr.start()

    for skill in skill_seq:              # internal steps go through patched env.step
        skill(env)

    audio_arr = mgr.stop_and_save("output/audio_0.wav")
    mgr.detach_from_env(env)             # restore original env.step
"""

from __future__ import annotations

import glob
import json
import os
import random

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Resolve paths relative to this file
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_HRTF_PATH   = os.path.join(_HERE, "hrtf",   "p0001.sofa")
_SRIR_DB_DIR = os.path.join(_HERE, "TAU_SRIR", "TAU-SRIR_DB")
_SNOISE_DIR  = os.path.join(_HERE, "TAU_SRIR", "TAU-SNoise_DB")
_SOUND_DIR   = os.path.join(_HERE, "sound")


# ---------------------------------------------------------------------------
# SRIR random selection
# ---------------------------------------------------------------------------

def _pick_random_srir() -> tuple[str | None, str | None]:
    """
    Randomly select one TAU-SRIR .mat file and its matching TAU-SNoise directory.

    Matching is done by the 2-digit numeric prefix shared between the filenames:
        rirs_04_pc226.mat  ↔  04_pc226_paatalo_office/

    Returns (srir_mat_path, snoise_dir).  Either can be None if the database
    directory is absent or empty.
    """
    mat_files = sorted(glob.glob(os.path.join(_SRIR_DB_DIR, "rirs_[0-9]*.mat")))
    if not mat_files:
        return None, None

    mat_path = random.choice(mat_files)
    # Extract the 2-digit prefix, e.g. "04" from "rirs_04_pc226.mat"
    basename = os.path.basename(mat_path)   # rirs_04_pc226.mat
    parts    = basename.split("_")          # ["rirs", "04", "pc226.mat"]
    prefix   = parts[1] if len(parts) > 1 else ""

    snoise_candidates = sorted(
        glob.glob(os.path.join(_SNOISE_DIR, f"{prefix}_*"))
    )
    snoise = snoise_candidates[0] if snoise_candidates else None
    return mat_path, snoise


def _resolve_sound_path(sound_file: str) -> str:
    """
    Resolve a sound_file entry from the JSON config.

    Resolution order:
      1. If absolute and exists – use directly.
      2. Relative to audio_generation/sound/.
      3. Relative to project root.
    """
    if os.path.isabs(sound_file) and os.path.exists(sound_file):
        return sound_file
    candidate = os.path.join(_SOUND_DIR, sound_file)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(_ROOT, sound_file)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"Sound file not found: '{sound_file}'. "
        f"Searched in {_SOUND_DIR}/ and {_ROOT}/"
    )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class AudioSimManager:
    """
    Manages binaural audio synthesis for a single AVLABench episode.

    Parameters
    ----------
    config_path : str
        Path to scene_audio_config.json.
    task_name : str
        Task name (key in config["tasks"]).
    sample_rate : int
        Output sample rate, default 44100 Hz.
    block_size : int
        Sounddevice callback block size, default 512 samples.

    Raises
    ------
    ValueError
        If *task_name* is not present in the config file.
    """

    def __init__(
        self,
        config_path: str | None,
        task_name: str,
        sample_rate: int = 44100,
        block_size: int = 512,
        config_dict: dict | None = None,
    ):
        # ------------------------------------------------------------------
        # Load config (from dict if provided, otherwise from file)
        # ------------------------------------------------------------------
        if config_dict is not None:
            cfg = config_dict
        else:
            with open(config_path, "r") as f:
                cfg = json.load(f)

        self.task_name = task_name
        self.cam_id    = int(cfg.get("cam_id", 0))

        # Allow the config to specify a custom HRTF path (absolute or relative
        # to the config file's directory).  Falls back to the default p0001.sofa.
        _cfg_hrtf = cfg.get("hrtf_path", None)
        if _cfg_hrtf is not None:
            if not os.path.isabs(_cfg_hrtf):
                _cfg_hrtf = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                         _cfg_hrtf)
            hrtf_path = _cfg_hrtf
        else:
            hrtf_path = _HRTF_PATH
        print(f"[audio] HRTF  : {os.path.basename(hrtf_path)}")

        task_cfg = cfg.get("tasks", {}).get(task_name)
        if task_cfg is None:
            raise ValueError(
                f"Task '{task_name}' not found in {config_path}. "
                f"Available tasks: {list(cfg.get('tasks', {}).keys())}"
            )
        self._sources_cfg: list[dict] = task_cfg["sources"]

        # ------------------------------------------------------------------
        # Select SRIR + SNoise randomly
        # ------------------------------------------------------------------
        srir_path, snoise_dir = _pick_random_srir()
        if srir_path:
            print(f"[audio] SRIR  : {os.path.basename(srir_path)}")
        else:
            print("[audio] SRIR  : none (TAU-SRIR_DB not found)")
        if snoise_dir:
            print(f"[audio] SNoise: {os.path.basename(snoise_dir)}")
        else:
            print("[audio] SNoise: none")

        # ------------------------------------------------------------------
        # Build BinauralAudioEngine  (imported here to avoid top-level cost)
        # ------------------------------------------------------------------
        from audio_generation.binaural_engine import BinauralAudioEngine  # noqa: E402
        self._engine_cls = BinauralAudioEngine   # kept for typing reference

        self.engine = BinauralAudioEngine(
            hrtf_path     = hrtf_path,
            sample_rate   = sample_rate,
            block_size    = block_size,
            srir_mat_path = srir_path,
            snoise_dir    = snoise_dir,
        )

        # ------------------------------------------------------------------
        # Register audio sources
        # ------------------------------------------------------------------
        for src in self._sources_cfg:
            sound_path = _resolve_sound_path(src["sound_file"])
            loop = bool(src.get("loop", True))
            self.engine.add_source(src["object_name"], sound_path, loop=loop)

        self._orig_step = None   # saved by attach_to_env
        # Counts env.step calls so per-source `start_step` gating
        # (used by audio-cued tasks like take_out_microwave_food) can hold
        # a source silent until the cue should fire.
        self._step_count = 0

    # ------------------------------------------------------------------
    # Env integration – monkey-patch env.step
    # ------------------------------------------------------------------

    def attach_to_env(self, env) -> None:
        """
        Replace env.step with a wrapper that calls _update_positions after
        every simulation step.  Call detach_from_env() to restore.
        """
        self._log_scene_sources(env)

        mgr  = self
        orig = env.step

        def _patched_step(action=None):
            result = orig(action)
            mgr._update_positions(env)
            return result

        self._orig_step = orig
        env.step        = _patched_step

    def _log_scene_sources(self, env) -> None:
        """Print which configured audio sources are present/absent in this scene."""
        scene_entities = set(env.task.entities.keys())
        present = [s for s in self._sources_cfg if s["object_name"] in scene_entities]
        absent  = [s for s in self._sources_cfg if s["object_name"] not in scene_entities]

        sep = "=" * 52
        print(f"\n[audio] {sep}")
        print(f"[audio]  Task : {self.task_name}")
        print(f"[audio]  Active sources ({len(present)}):")
        for src in present:
            print(
                f"[audio]    • {src['object_name']:28s} "
                f"→ {src['sound_file']}  (gain={src.get('gain', 1.0)})"
            )
        if absent:
            print(f"[audio]  Not in scene ({len(absent)}):")
            for src in absent:
                print(f"[audio]    ✗ {src['object_name']}")
        print(f"[audio] {sep}\n")

    def detach_from_env(self, env) -> None:
        """Restore the original env.step."""
        if self._orig_step is not None:
            env.step        = self._orig_step
            self._orig_step = None

    # ------------------------------------------------------------------
    # Per-step spatial update
    # ------------------------------------------------------------------

    def _update_positions(self, env) -> None:
        """
        Read the main camera pose from physics and update each source's HRTF.

        Camera pose in MuJoCo
        ---------------------
        physics.data.cam_xpos[cam_id]  – world position (3,)
        physics.data.cam_xmat[cam_id]  – world rotation, row-major 9-vector
            col 0 → right, col 1 → up, col 2 → backward  (MuJoCo convention)

        For each registered sound source:
          1. Get world position of the named body/geom.
          2. Compute (azimuth, elevation, distance) in the camera frame.
          3. Interpolate HRTF and update the source with combined gain
             (distance attenuation × user-specified gain from JSON).
        """
        # Per-step counter drives the `start_step` gate on each source so
        # delayed cues (e.g. the microwave chime) stay silent until N env.step
        # calls have elapsed. We increment unconditionally so that a missing
        # cam pose can never desync the count from the rollout.
        self._step_count += 1

        try:
            cam_pos  = env.physics.data.cam_xpos[self.cam_id].copy()   # (3,)
            cam_xmat = env.physics.data.cam_xmat[self.cam_id].copy()   # (9,)
        except Exception:
            return

        # Import here to keep the module lightweight at load time
        from audio_generation.binaural_engine import compute_listener_relative_direction  # noqa: E402

        for src in self._sources_cfg:
            name      = src["object_name"]
            geom_type = src.get("geom_type", "body")
            user_gain = float(src.get("gain", 1.0))
            start_step = int(src.get("start_step", 0))

            # Keep the source silent (no update_hrtf → SoundSource._active stays
            # False) until the configured step has been reached.
            if self._step_count < start_step:
                continue

            try:
                if geom_type == "body":
                    # composer namespaces bodies, so find("body", name) returns None.
                    # Use the entity's worldbody binding directly instead.
                    entity = env.task.entities.get(name)
                    if entity is None:
                        continue
                    obj_pos = entity.get_xpos(env.physics).copy()
                else:
                    obj_pos = env.get_xpos_by_name(name, geom_type).copy()
            except Exception:
                # Object may not exist in this episode; skip silently
                continue

            az, el, dist = compute_listener_relative_direction(
                cam_pos, cam_xmat, obj_pos
            )

            if name not in self.engine.sources:
                continue

            hrir_l, hrir_r = self.engine._interpolator.interpolate(az, el)
            dist_gain      = 1.0 / max(dist, 0.05)
            total_gain     = dist_gain * user_gain
            self.engine.sources[name].update_hrtf(hrir_l, hrir_r, total_gain)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Clear the recording buffer and begin audio playback."""
        self.engine.clear_recording()
        self.engine.start()

    def stop_and_save(self, wav_path: str) -> np.ndarray:
        """
        Stop the audio stream, write the recorded binaural audio to *wav_path*,
        and return the audio array as float32 [N_samples, 2].

        The parent directory of *wav_path* is created if it does not exist.
        """
        self.engine.stop()
        audio = self.engine.get_recorded_audio()   # [N, 2] float32

        if audio.shape[0] > 0:
            os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
            sf.write(wav_path, audio, self.engine.sr)
            duration = audio.shape[0] / self.engine.sr
            print(f"[audio] Saved {duration:.2f}s binaural audio → {wav_path}")
        else:
            print("[audio] WARNING: no audio recorded (empty buffer)")

        return audio
