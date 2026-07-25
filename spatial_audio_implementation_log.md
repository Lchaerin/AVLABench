# Spatial-audio SmolVLA — implementation log

Running notes from the session that built Path-A audio-conditioned SmolVLA on
top of `lerobot/smolvla_vlabench` for the `select_radio` task. Pairs with
`spatial_audio_vla_pipeline.md` (design doc) — this file captures *what was
actually built*, *what broke along the way*, and *what still needs doing*.

## 1. Goal recap

- Fine-tune SmolVLA so it can disambiguate which radio to press from
  **binaural audio** (SLED predicts class + DOA, VLA decides which button).
- Fixed English instruction for every episode:
  `"primitive: Press the button in front of the radio that is making sound."`
- Base checkpoint: **`lerobot/smolvla_vlabench`** (trained on
  `lerobot/vlabench_unified`, 3 cameras × 224² + 7-dim state + 7-dim action).
- Audio injection = **Path A only** (SLED events as LLM-readable tokens with
  direction embedding at each `@` slot). Path C (vision-audio cross-attn)
  left for later.

## 2. Directory layout created this session

```
src/
├── audio/
│   ├── direction_encoder.py        # (az, el) → hidden_dim MLP, std=0.01 init
│   ├── audio_token_builder.py      # "[AUDIO] cls @ conf 0.NN ; ... [/AUDIO]"
│   └── sled_features.py            # SLED JSON → [T, K] dense + temporal smoothing
├── data/
│   └── convert_hdf5_to_lerobot.py  # HDF5 → LeRobot dataset (vlabench_unified schema)
├── models/
│   └── smolvla_audio.py            # AudioAwareSmolVLAPolicy (Path A injection)
├── training/
│   └── train_smolvla_audio.py      # fine-tune loop (bf16 autocast, split LR)
├── eval/
│   ├── eval_smolvla_audio.py       # closed-loop evaluation with realtime SLED+VLA
│   ├── show_audio_to_vla.py        # one-shot SLED → VLA tensor demo
│   └── dump_vla_inputs.py          # full per-step text dump for one episode
└── README.md                       # end-to-end usage
sh/
├── test_smolvla_audio.sh           # generate → convert → train orchestrator
└── eval_smolvla_audio.sh           # eval wrapper
```

## 3. Architecture notes

`AudioAwareSmolVLAPolicy` wraps the stock `SmolVLAPolicy` with one change:
the prefix now includes an AUDIO block between language and state.

```
prefix = [image embs]           3 cams × 64 SigLIP tokens   = 192 vectors
       + [language embs]        tokenizer("primitive: Press … \n")
       + [AUDIO embs]           <- new; 38–64 vectors depending on content
       + [state emb]            state_proj([x, y, z, rx, ry, rz, gripper])
```

Inside the AUDIO block:

1. `AudioTokenBuilder.build_batch` formats K SLED events as
   `"[AUDIO] <class_words> @ conf 0.NN ; ... [/AUDIO]"`, where `@` is the
   *direction placeholder* that later gets its embedding perturbed.
   Empty slots get `silence @ conf 0.00`.
2. The VLM's text embedding layer turns those ids into hidden vectors.
3. `DirectionEncoder` (sin/cos → 2-layer MLP → hidden_size) maps (az, el)
   per slot to a hidden vector and **adds** it on top of the embedding at
   each `@` slot. Last-layer init `std=0.01` keeps the policy close to its
   pretrained behaviour on step 0.

## 4. Dataset format — matches `lerobot/vlabench_unified`

| key                                | shape        | notes                                 |
|------------------------------------|--------------|---------------------------------------|
| `observation.images.image`         | (224,224,3)  | front camera (cam_2)                  |
| `observation.images.second_image`  | (224,224,3)  | right camera (cam_0)                  |
| `observation.images.wrist_image`   | (224,224,3)  | wrist cam (cam_3)                     |
| `observation.state`                | (7,)         | `[x, y, z, rx, ry, rz, gripper]`      |
| `action`                           | (7,)         | same layout, `gripper_cmd ∈ {0, 1}`   |
| `observation.audio.azimuth_deg`    | (top_k,)     | top-K SLED events (new)               |
| `observation.audio.elevation_deg`  | (top_k,)     | "                                     |
| `observation.audio.confidence`     | (top_k,)     | "                                     |
| `observation.audio.class_id`       | (top_k,)     | -1 means empty slot                   |

Conversion rules applied by `src/data/convert_hdf5_to_lerobot.py`:

- 8-dim action `[x,y,z,rx,ry,rz,g_left,g_right]` → 7-dim
  `[x,y,z,rx,ry,rz,binarized_gripper]` (threshold 0.02).
- 4 cameras (480×480) → 3 cameras (224×224) via `cv2.resize`.
- Empty SLED frames are **forward-filled** with linearly-decayed confidence
  for up to ~2.5 s (25 frames @ 10 fps), matching the SLED window length.
- Task string is overridden to
  `"primitive: Press the button in front of the radio that is making sound."`
  regardless of the underlying episode's position label — audio becomes
  the only disambiguating signal.

## 5. Environment / dependency gotchas

| Symptom                                                                                                   | Fix                                                     |
|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `TypeError: non-default argument 'backbone_cfg' follows default argument` at SmolVLA import              | Pin `transformers>=4.57.1,<5.0.0` (lerobot 0.4.4 needs ≤5.0) |
| `ImportError: Package 'num2words' is required`                                                            | `pip install num2words`                                 |
| `FileExistsError` in `LeRobotDataset.create`                                                              | Don't pre-create out_dir; converter handles it          |
| `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda for BFloat16`                        | Drop `GradScaler` (it's fp16-only; bf16 has dynamic range) |
| `ValueError: At least one stride in the given numpy array is negative` on mujoco render                  | Wrap `env.physics.render(...)` with `np.ascontiguousarray` |
| Audio recording stuck at 0.13 s even after N env steps                                                    | Replace env-step warmup with wall-clock `time.sleep(s)` — the engine callback runs at wall-clock rate |
| Pretrained checkpoint expects 3 cams + state/action dim 6/7                                               | `policy.config.input_features` / `output_features` are overridden post-load to match our data |

## 6. Training recipe (verified working)

```bash
# one-time setup
conda activate vlabench
pip install "transformers>=4.57.1,<5.0.0" accelerate num2words

# generate → convert → fine-tune
N_SAMPLE=400 TRAIN_STEPS=20000 BATCH_SIZE=32 NUM_WORKERS=8 \
    bash sh/test_smolvla_audio.sh
```

Smoke test (27 episodes, 3 optimizer steps) loss curve:
```
step 1: loss=0.89   step 2: loss=1.13   step 3: loss=1.12
[trainable] 100.02M / 450.19M params
≈ 2.5 step/s at batch=2 on RTX 5090, 480×480→224×224 images
```

Hyperparameters recommended for RTX 5090, single-task select_radio:

| setting         | smoke     | quick sanity (~3 h) | full (~10 h) |
|-----------------|-----------|----------------------|--------------|
| episodes        | 30        | 150                  | 400          |
| batch size      | 2         | 32                   | 32 (64 with care) |
| train steps     | 5         | 5 000                | 15 000–20 000 |
| LR (expert)     | 1e-4      | 5e-5                 | 5e-5         |
| LR (audio)      | 1e-4      | 1e-4                 | 1e-4         |
| warmup steps    | 1         | 250                  | 500          |
| num_workers     | 0         | 8                    | 8            |

## 7. Timing model — SLED vs VLA sync

Both run **concurrently**; their clocks are decoupled:

| component              | rate         | role                                           |
|------------------------|--------------|------------------------------------------------|
| `sounddevice` callback | 44 100 Hz    | binaural engine fills `_rec_buf` in real time  |
| SLED daemon thread     | 5 Hz (200 ms)| reads last **480 ms** (v5) of buffer, updates snapshot under a lock |
| VLA inference loop     | 10–20 Hz     | `get_latest_prediction()` → build batch → predict chunk |

Audio window fed to the VLA per query = **480 ms** (SLED v5; 960 ms for v3/v4).
The VLA sees the empty snapshot (`class_id=-1`) during the first ~1 s before
SLED has produced anything, or whenever SLED's threshold rejects a frame —
matches the dropout pattern it was trained under.

## 8. Evaluation

```bash
# latest ckpt in outputs/smolvla_audio/, 20 episodes, videos + audio log
bash sh/eval_smolvla_audio.sh

# specific ckpt, no video, 50 episodes
CKPT=outputs/smolvla_audio/ckpt_step0010000.pt SAVE_VIDEO=0 N=50 \
    bash sh/eval_smolvla_audio.sh
```

The eval script:
- Instantiates one `SLEDOverlay(realtime=True)` and reuses it across episodes
  (rebinding `set_audio_engine` each reset).
- Pulls a fresh snapshot from SLED on every VLA step via
  `get_latest_prediction()` (μs cost, just a lock + copy).
- Applies 7-dim policy actions via IK + `env.step(9-dim qpos+gripper)`, using
  `HORIZON` actions per chunk before re-predicting (default 5).
- Writes `eval_summary.json`, `eval_infos.json` (with per-episode
  `sled_confident_rate`, `sound` metadata, final audio snapshot), optional
  `audio/audio_eval_*.wav`, `audio_logs/audio_log_ep*.json` (per-step SLED
  snapshot trace), and per-episode MP4 visualisations.

## 9. Auxiliary scripts

- **`src/eval/show_audio_to_vla.py`** — one-off walkthrough: hand-crafted
  SLED snapshot → token IDs → direction encoder output → audio embedding
  Δ at each slot, plus a prose description of the sync model. Useful for
  debugging "is the direction encoder actually doing anything".

- **`src/eval/dump_vla_inputs.py`** — run one real episode and write every
  text the VLA sees, step by step, to `outputs/vla_inputs/episode_<N>.txt`.
  Covers the constant instruction (token IDs + decoded pieces) plus the
  per-step audio block (token IDs, direction-slot markers, full combined
  LLM-stream text).

## 10. Dataset-quality gate (new CLI in trajectory_generation.py)

```bash
# defaults: drop episode if fewer than 40% of SLED frames had top-1 conf ≥ 0.30
python scripts/trajectory_generation.py \
    --task-name select_radio \
    --audio-config audio_generation/scene_audio_config.json \
    --sled-ckpt /home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt \
    --save-dir ./dataset_v5 \
    --n-sample 400 \
    --sled-min-confident-rate 0.4 \
    --sled-conf-thresh 0.30
```

When the gate fires, the dataset video, WAV, viz video, SLED JSON, GT JSON,
audio meta, and HDF5 are all skipped / cleaned up — the episode never makes
it into training.

## 11. ⚠️ Critical bug discovered & fixed: direction slot detection

`AudioTokenBuilder._single_token_id("@")` returned the standalone-`@` id
(**48**), but inside the actual audio text the placeholder is tokenised
with a leading space (`Ġ@` = **3394**). Result: `dir_slot_mask` was always
all-False and **the direction encoder output never reached the LLM during
the 10 000-step training run**. The fix:

```python
def _collect_placeholder_token_ids(self, ch):
    """Union of every single-token id that decodes to `ch` across surface forms."""
    ids = set()
    for surface in (ch, " " + ch, ch + " ", " " + ch + " "):
        for tid in self.tokenizer.encode(surface, add_special_tokens=False):
            if self.tokenizer.decode([tid]).strip() == ch:
                ids.add(int(tid))
    return sorted(ids)
```

After the fix, verification on the real tokenizer:
```
dir_token_ids: [48, 3394]
@ slot positions: [7, 16, 25]
per-k positions: [[7], [16], [25]]
```

**Implication**: the current `ckpt_step0010000.pt` saw class-name text
(`wind brass`, `speech`, …) but zero direction information. It's equivalent
to the "Text-only audio" baseline in `spatial_audio_vla_pipeline.md`.
To actually test Path-A spatial audio, **retrain with the fixed builder** —
no data regeneration required.

## 12. Demonstration of SLED → VLA flow

From `src/eval/dump_vla_inputs.py`, episode 0, step 0 (GT sound = Wind_Brass;
SLED misclassifies as Piano_Keys — realistic noisy example):

```
SLED snapshot (top-3):
    slot 0: cid=  8  class=Piano_Keys    az=  +4.82°  el= -23.20°  conf=0.964
    slot 1: cid= -1  class=—             az=  +0.00°  el=  +0.00°  conf=0.000
    slot 2: cid= -1  class=—             az=  +0.00°  el=  +0.00°  conf=0.000

Audio block text:
    [AUDIO] piano keys @ conf 0.96 ; silence @ conf 0.00 ; silence @ conf 0.00 [/AUDIO]

Tokenised audio (38 tokens, ★ = direction slot):
    [pos  5] id=15717  tok='Ġpiano'
    [pos  6] id= 8889  tok='Ġkeys'
    [pos  7] id= 3394  tok='Ġ@' ★ slot k=0
    [pos  8] id= 1086  tok='Ġconf'
    ...
    [pos 16] id= 3394  tok='Ġ@' ★ slot k=1
    ...
    [pos 25] id= 3394  tok='Ġ@' ★ slot k=2

Combined LLM-stream text (instruction → audio):
    primitive: Press the button in front of the radio that is making sound.
    [AUDIO] piano keys @ conf 0.96 ; silence @ conf 0.00 ; silence @ conf 0.00 [/AUDIO]

[VLA timing] 1124 ms for the first chunk (includes model warm-up),
             ~60–100 ms steady-state on RTX 5090.
```

## 13. Known open items

1. **Retrain with fixed token builder** — required before any conclusion
   about whether spatial audio helps vs a text-only baseline.
2. **Action/state normalization is bypassed** — the custom loop skips
   LeRobot's `NormalizerProcessorStep`, so MSE is in raw units. Either
   compute dataset stats and normalize manually, or wire in
   `make_smolvla_pre_post_processors`.
3. **Pretrained action-dim mismatch partially absorbed by padding** — our
   7-dim matches `lerobot/vlabench_unified`, but the normalizer's saved
   stats are for that dataset's specific joint-angle distribution. Fresh
   stats from the AVLABench trajectories would improve convergence speed.
4. **Real-time SLED sync** works under realtime=True + `set_audio_engine`;
   consider bumping `infer_hz` from 5 to 10 once we're confident about
   the gpu budget (each inference is ~10 ms on RTX 5090).
5. **Path C (vision-audio cross-attention)** is not implemented. Adding it
   would require extending `embed_prefix_with_audio` with the adapter
   described in `spatial_audio_vla_pipeline.md` §4.
