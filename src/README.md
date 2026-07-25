# Spatial-audio fine-tuning of SmolVLA on AVLABench

This directory wires the `spatial_audio_vla_pipeline.md` design into
SmolVLA + the AVLABench `select_radio` task.

```
src/
├── audio/
│   ├── direction_encoder.py     # (az, el) -> hidden_dim embedding (sin/cos MLP)
│   ├── audio_token_builder.py   # SLED top-K events -> language tokens + dir slots
│   └── sled_features.py         # parse + temporally-smooth sled_predictions JSON
├── data/
│   └── convert_hdf5_to_lerobot.py   # AVLABench HDF5 -> LeRobot dataset format
├── models/
│   └── smolvla_audio.py             # AudioAwareSmolVLAPolicy (Path A injection)
└── training/
    └── train_smolvla_audio.py       # fine-tuning entry-point
```

## End-to-end pipeline

### 0. Prerequisites

The training entry-point uses `lerobot[smolvla]` which depends on the
HuggingFace `transformers` library plus a couple of small extras pulled in
by SmolVLM2's processor. Install once into the `vlabench` env (verified
versions in parens):

```bash
conda activate vlabench
pip install "transformers>=4.57.1,<5.0.0"   # lerobot 0.4.4 needs <5.0
pip install accelerate num2words
# (already present: lerobot==0.4.4, torch==2.10, datasets, safetensors)
```

> Note: `transformers>=5.0` (and the `huggingface-hub>=1.0` it pulls in)
> breaks `lerobot 0.4.4`'s GR00T import chain — you'll see
> `TypeError: non-default argument 'backbone_cfg' follows default argument`
> on `from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy`.
> Pin to `4.57.x` for now.

### 1. Generate trajectories with binaural audio + SLED predictions

```bash
python scripts/trajectory_generation.py \
    --task-name select_radio \
    --audio-config audio_generation/scene_audio_config.json \
    --sled-ckpt /home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt \
    --save-dir ./dataset_v5 \
    --n-sample 200          # > 15: aim for ~150 successful episodes
```

Each successful episode produces:
- `dataset_v5/select_radio/data_<i>.hdf5`         (RGB + state + action + SLED JSON)
- `dataset_v5/select_radio/audio_<i>.wav`         (resampled binaural recording)
- `dataset_v5/select_radio/audio_meta_<i>.json`   (active radio + sound metadata)
- `dataset_v5/select_radio_gt/gt_<i>.json`        (ground-truth DOA + class)
- `dataset_v5/select_radio_viz/demo_<i>_*.mp4`    (markers + audio for inspection)

### 2. Convert HDF5 → LeRobot dataset

The converter overrides the per-episode instruction with a **fixed English**
prompt — `"Press the button in front of the radio that is making sound."` —
so the model can only solve the task by attending to the audio modality.

```bash
python src/data/convert_hdf5_to_lerobot.py \
    --src-dir   ./dataset_v5/select_radio \
    --out-dir   ./dataset_v5_lerobot \
    --repo-id   local/avla_select_radio \
    --fps       10 \
    --top-k     3
```

This writes a LeRobot v3.0 dataset under `dataset_v5_lerobot/` whose
feature schema mirrors **`lerobot/vlabench_unified`** (the dataset that
`lerobot/smolvla_vlabench` was trained on), so the pretrained
state/action projections, SigLIP camera embeddings and normalisation
statistics stay meaningful at fine-tune time:

| key                                | shape         | notes                          |
|------------------------------------|---------------|--------------------------------|
| `observation.images.image`         | (224, 224, 3) | front camera (cam_2)           |
| `observation.images.second_image`  | (224, 224, 3) | right camera (cam_0)           |
| `observation.images.wrist_image`   | (224, 224, 3) | wrist camera (cam_3)           |
| `observation.state`                | (7,)          | `[x, y, z, rx, ry, rz, gripper]` |
| `action`                           | (7,)          | same layout, `gripper_cmd ∈ {0, 1}` |
| `observation.audio.azimuth_deg`    | (3,)          | top-K SLED events (*new*)      |
| `observation.audio.elevation_deg`  | (3,)          | "                              |
| `observation.audio.confidence`     | (3,)          | "                              |
| `observation.audio.class_id`       | (3,)          | -1 means empty slot            |

Task prefix: **`"primitive: Press the button in front of the radio that is making sound."`**
— matches the `"primitive: …"` convention used by `vlabench_unified`.

Our 8-dim raw HDF5 action `[x, y, z, rx, ry, rz, g_left, g_right]` is
collapsed to 7-dim by binarising the two finger positions (threshold
0.02): open ↔ 1.0, closed ↔ 0.0. State reuses the same pose, giving a
close stand-in for the "executed" pose convention used by
vlabench_unified at 10 fps.

Frames where SLED returned no events get the most recent prediction
forward-filled (with linearly decayed confidence) for up to ~2.5 s.

The cameras are remapped on the fly by the converter — default:
`--cam-image 2 --cam-second-image 0 --cam-wrist-image 3` — flip these
if your VLABench XML enumerates cameras differently.

### 3. Fine-tune SmolVLA + audio modules

Loads `lerobot/smolvla_vlabench`, wraps it with a Path-A audio block
(directional embedding injected into the language stream), and trains the
**action expert + audio modules** while keeping SigLIP and the LLM frozen.

```bash
python src/training/train_smolvla_audio.py \
    --dataset-root  ./dataset_v5_lerobot \
    --pretrained    lerobot/smolvla_vlabench \
    --taxonomy      ./class_taxonomy.yaml \
    --output-dir    ./outputs/smolvla_audio_v0 \
    --batch-size    8 \
    --steps         5000 \
    --lr            1e-4 \
    --audio-lr      2e-4 \
    --num-workers   4
```

Optional alignment-only first phase (only the direction encoder learns):

```bash
python src/training/train_smolvla_audio.py ... \
    --train-audio-only --steps 500 --output-dir ./outputs/smolvla_audio_align
```

Then re-launch full training using the Phase-0 checkpoint.

## Architecture summary

`AudioAwareSmolVLAPolicy` = stock `SmolVLAPolicy` with a single change inside
`embed_prefix`:

```
prefix = [image embs] + [language embs] + [AUDIO embs] + [state emb]
```

`AUDIO embs` is built per-sample:

1. `AudioTokenBuilder` formats the SLED top-K events as
   `"[AUDIO] cls0 @ conf 0.NN ; cls1 @ conf 0.NN ; cls2 @ conf 0.NN [/AUDIO]"`
   (literal `silence @ conf 0.00` for empty slots).
2. The VLM's text embedding layer turns the ids into hidden vectors.
3. `DirectionEncoder` maps each (az, el) pair to a hidden-size vector and
   *adds* it on top of the embedding at the corresponding `@` slot. The MLP's
   final layer is initialised at `std=0.01`, so on step 0 the audio block is
   essentially pure-text and the policy stays close to its pretrained
   behaviour.

This is "Path A" of the spatial-audio pipeline doc — the simplest
extension that lets the VLM reason about *what* sound is *where* using its
existing language capabilities. Path C (vision-audio cross-attention) can
be layered on later by adapting `embed_prefix_with_audio` in
`smolvla_audio.py`.

## Smoke test results (2026-04-23)

3-step end-to-end fine-tuning of `lerobot/smolvla_vlabench` on 6 episodes
ran cleanly on RTX 5090:

```
[trainable] 100.02M / 450.19M params
[data] num samples = 420 (6 episodes), chunk_size=50
step 1: loss=1.0790    step 2: loss=0.4149    step 3: loss=0.9373
```

≈ 2.5 s / step at batch_size=2, single 480×480 front camera, bf16 autocast.

### Known gaps to fix before serious training

1. **No action/state normalization.** The custom training loop bypasses
   LeRobot's `NormalizerProcessorStep`, so MSE losses are in raw action
   units. Either (a) compute dataset stats and apply mean/std normalization
   manually, or (b) wire in `make_smolvla_pre_post_processors` from
   `lerobot.policies.smolvla.processor_smolvla`.
2. **Pretrained checkpoint mismatch.** The `lerobot/smolvla_vlabench`
   checkpoint expects 3 cameras at (3, 256, 256), state_dim=6, action_dim=7.
   We override `policy.config.input_features` and `output_features` after
   loading (training script lines around `from_pretrained_with_audio`),
   which makes `prepare_images` look for our single `observation.images.image`
   — but the embedding dimensions inside the model (max_state_dim=32,
   max_action_dim=32) absorb the size difference because both sides are
   padded to 32.
3. **`add_image_special_tokens=False`** in the pretrained config means our
   prefix is `[image embs] + [language embs] + [audio embs] + [state emb]`
   without any `[FAKE_IMAGE]/[GLOBAL_IMAGE]` markers. If you ever flip
   that flag in the config, also update `embed_prefix_with_audio` to mirror
   the markers between language and audio if you want them.
