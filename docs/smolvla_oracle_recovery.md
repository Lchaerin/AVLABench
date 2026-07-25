# SmolVLA Oracle Audio Recovery Notes

This note records the recovery experiments for the SmolVLA audio/oracle setup.
Large checkpoints and raw `outputs/` artifacts are not committed because the
repository ignores generated outputs and the checkpoint files are very large.

## Branch State

- Working branch used for recovery: `recover-smolvla-audio`
- `main` and `recover-smolvla-audio` were aligned at the time this note was
  created.
- The workspace still contains uncommitted generated artifacts, including
  checkpoints, datasets, and ignored output directories.

## Code Changes Already Recorded

- Added train-time augmentation options and resume support for SmolVLA audio
  training.
- Added checkpoint-aware eval loading so the evaluator prefers the checkpoint's
  saved `audio_config`.
- Added an alternate `class_tokens` audio fusion mode while preserving the
  original inline fusion path for checkpoint compatibility.

## 100-Episode Oracle Eval Results

All numbers below are from `eval_summary.json` files under `outputs/`.

| Run | Success rate | Notes |
| --- | ---: | --- |
| `eval_smolvla_oracle_ver1_recover_100` | 0.01 | Re-eval of old ver1 checkpoint in current code/runtime |
| `eval_smolvla_oracle_class_tokens` | 0.28 | New class-token fusion baseline |
| `eval_smolvla_oracle_class_tokens_lm4` | 0.32 | Class-token fusion with more LM layers unfrozen |
| `eval_smolvla_oracle_ver2` | 0.33 | Inline oracle variant |
| `eval_smolvla_oracle_ver3` | 0.42 | Best result before resume chain |
| `eval_smolvla_oracle_ver4` | 0.37 | Resumed from ver3 |
| `eval_smolvla_oracle_ver5` | 0.47 | Best completed result so far |
| `eval_smolvla_oracle_ver6` | 0.45 | Continued tuning |
| `eval_smolvla_oracle_ver7` | 0.46 | Continued tuning |

Current best completed 100-episode oracle result is `ver5` at 47%, below the
target success rate of 60%.

## Two-Radio Oracle Eval Results

Task: `select_radio_two`, launched through
`sh/train_smolvla_oracle_two_radios.sh`. Two radios emit different sound
classes; the instruction names the target sound class.

### Two-Radio Changes Tried

The key constraint for the preferred benchmark is that SmolVLA must still
generate the action trajectory. SELD/audio preprocessing is allowed, but the
evaluation path must not directly choose a button or synthesize a trajectory.

Implementation details that matter for reproducing the current state:

- Training audio perturbation lives in
  `src/training/train_smolvla_audio.py::_apply_oracle_noise`.
  It accepts the per-frame oracle audio arrays:
  `class_id`, `azimuth_deg`, `elevation_deg`, and `confidence`.
- Eval-time audio snapshots are built in
  `src/eval/eval_smolvla_audio.py::evaluate_episode`.
  For oracle eval, `_oracle_snapshot` builds the top-k audio dictionary from
  simulator ground truth plus noise. For real SLED, `_live_snapshot` converts
  SLED output into the same dictionary shape.
- The policy still receives a normal batch and calls
  `policy.predict_action_chunk(batch)`. No controller-level shortcut is in
  the valid runs.

Slot handling changes:

- `--shuffle-slots on` randomizes present audio slots during training after
  noise injection. This breaks the dataset convention where slot 0 contains
  the target source, so the model cannot solve the two-radio task by always
  reading slot 0.
- `--shuffle-audio-slots on` applies the same idea at eval time, reflecting
  the user's requirement that multi-source SELD output order should be random
  apart from noise.
- `target_slot_protect=on` prevents oracle class-flip augmentation from
  corrupting the dataset's original target slot before shuffling. Without this,
  a training example can become ambiguous because the instruction's target
  sound class may disappear from all slots.
- Class smoothing is applied before eval-time slot shuffling. This is important
  because smoothing is per physical slot/source history; smoothing after a
  random shuffle can pair a class ID from one source with another source's
  azimuth.
- `--target-first-slots` and `--target-first-audio-slots` were added as a
  target-aware ablation. They move the slot whose class matches the
  instruction target to slot 0. This does not bypass SmolVLA trajectory
  generation, but it uses instruction-target knowledge before the policy sees
  the input, so it is no longer considered the preferred benchmark.
- `--canonicalize-slots azimuth` and
  `--canonicalize-audio-slots azimuth` are the current target-agnostic
  alternative. They sort present detections by reported SELD azimuth from
  task-left to task-right and do not inspect the instruction or target class.
  The implementation is in
  `src/training/train_smolvla_audio.py::_canonicalize_slots_by_azimuth` and
  `src/eval/eval_smolvla_audio.py::_canonicalize_audio_slots_by_azimuth`.

Audio fusion changes:

- `src/audio/audio_token_builder.py` now supports `natural_language=True`.
  In that mode each source becomes a readable sentence, for example:
  `sound 1: alarm is to the left @, azimuth 52 degrees, confidence 0.94.`
- `@` is a placeholder token. The tokenizer can encode variants such as `@`
  and ` @`, so `AudioTokenBuilder._collect_placeholder_token_ids` collects all
  single-token surface forms that decode back to `@`.
- `src/models/smolvla_audio.py::_embed_audio_block` replaces each located
  `@` token embedding with the corresponding `DirectionEncoder(az, el)` output
  gated by confidence. For `natural_language`, the continuous direction vector
  is therefore co-located with the class words and coarse spatial language in
  the same sentence.
- `class_tokens` remains available as an alternate fusion path. In that mode,
  the text block stays compact and learned per-class direction tokens are
  appended via `_embed_direction_tokens`.

Direction/sign convention changes:

- The initial natural-language implementation described positive azimuth as
  right, which was inconsistent with the task metadata. Eval logs showed that
  left-position targets have positive stored azimuth and right-position targets
  have negative stored azimuth.
- `src/audio/audio_token_builder.py::_azimuth_words` was changed so positive
  stored azimuth maps to `left`, negative maps to `right`, and small magnitude
  maps to `straight ahead`.

Wrapper/script changes:

- `sh/train_smolvla_oracle_two_radios.sh` exposes the two-radio knobs:
  `AUDIO_FUSION_MODE`, `SHUFFLE_SLOTS`, `TARGET_SLOT_PROTECT`,
  `TARGET_FIRST_SLOTS`, `CANONICALIZE_SLOTS`,
  `SHUFFLE_AUDIO_SLOTS_EVAL`, `TARGET_FIRST_AUDIO_SLOTS_EVAL`, and
  `CANONICALIZE_AUDIO_SLOTS_EVAL`.
- The same wrapper forwards regularization and adaptation knobs used in the
  experiments: `UNFREEZE_LM_LAYERS`, `VLM_LORA`, `LORA_TARGET_MODULES`,
  `IMAGE_COLOR_JITTER`, `IMAGE_TRANSLATE_PX`, `STATE_NOISE_STD`,
  `DIRECTION_DROPOUT`, and `AUDIO_CONF_DROPOUT`.
- `scripts/auto_smolvla_two_radio_nl_loop.sh` records the intended automatic
  progression through natural-language, LoRA, and class-token variants. In
  practice, the current long-running `v5` process was started directly with
  `setsid` because plain `nohup ... &` was not keeping the child process alive
  reliably in this shell environment.

Invalid shortcut result:

- The attempted `audio_button` controller result is treated as invalid for the
  SmolVLA policy benchmark because it bypassed learned trajectory generation.
  It was removed from the accepted path. Keep it only as a sanity check that
  the audio target can be identified when trajectory generation is not the
  bottleneck.

### Completed 100-Episode Results

| Run | Success rate | Notes |
| --- | ---: | --- |
| `eval_smolvla_oracle_two_ct_lm4_v1_step4k` | 0.05 | Learned policy, 4k checkpoint, shuffled slots |
| `eval_smolvla_oracle_two_ct_lm4_v1_step4k_fixsmooth` | 0.04 | Learned policy after smoothing-order fix |
| `eval_smolvla_oracle_two_audio_button_100` | 0.92 | Invalid for policy benchmark: bypassed SmolVLA trajectory generation |
| `eval_smolvla_oracle_two_targetfirst_40k_v1` | 0.47 | Learned-policy trajectory result, 40k checkpoint, but target-aware SELD canonicalization. This is now treated as an ablation, not the preferred benchmark. |
| `eval_smolvla_oracle_two_nl_40k_v1` | 0.28 | Non-target-first natural-language fusion. Exposed azimuth sign issue. |
| `eval_smolvla_oracle_two_nl_signfix_40k_v2` | 0.25 | Natural-language sign convention fixed, but still failed left side. |
| `eval_smolvla_oracle_two_nl_signfix_lm6_40k_v3` | 0.25 | Direction placeholder co-located inside each natural-language sentence; top 6 LM layers unfrozen. |
| `eval_smolvla_oracle_two_nl_signfix_lora_40k_v4` | 0.29 | Same co-located natural-language structure, VLM LoRA on `q_proj,v_proj`. |
| `eval_smolvla_oracle_two_nl_azcanon_lm6_40k_v5` | 0.29 | Target-agnostic azimuth canonicalization, natural-language fusion, top 6 LM layers unfrozen. Intention success was 0.43 but actual success stayed low. |
| `eval_smolvla_oracle_two_nl_azcanon_lora_40k_v6` | 0.47 | **Best preferred benchmark so far.** Target-agnostic azimuth canonicalization, natural-language fusion, VLM LoRA on `q_proj,v_proj`. Exceeds the 0.40 target without target-first canonicalization. |

Per-position breakdown for the non-target-first completed runs:

| Run | Left | Middle | Right | Main failure pattern |
| --- | ---: | ---: | ---: | --- |
| `eval_smolvla_oracle_two_nl_40k_v1` | 0/36 | 19/33 | 9/31 | Natural-language text had the wrong left/right sign convention. |
| `eval_smolvla_oracle_two_nl_signfix_40k_v2` | 0/36 | 17/33 | 8/31 | Sign fix alone did not make direction actionable. |
| `eval_smolvla_oracle_two_nl_signfix_lm6_40k_v3` | 0/36 | 23/33 | 2/31 | Co-located direction helped middle but collapsed side choices. |
| `eval_smolvla_oracle_two_nl_signfix_lora_40k_v4` | 0/36 | 26/33 | 3/31 | LoRA increased center bias instead of learning left/right. |
| `eval_smolvla_oracle_two_nl_azcanon_lm6_40k_v5` | 1/36 | 22/33 | 6/31 | Azimuth canonicalization improved right a little, but actual success remained 0.29. |
| `eval_smolvla_oracle_two_nl_azcanon_lora_40k_v6` | 9/36 | 21/33 | 17/31 | LoRA plus azimuth canonicalization substantially improved side-button success and cleared the 0.40 target. |

The target-first run exceeded the requested 40% target, but it uses the
instruction target class to reorder SELD slots before the policy sees them.
Because that is target-aware, it is now kept only as an ablation. Its
by-position breakdown was:

- left: 11/36, 30.6%
- middle: 28/33, 84.8%
- right: 8/31, 25.8%

The preferred benchmark keeps `TARGET_FIRST_SLOTS=off` and
`TARGET_FIRST_AUDIO_SLOTS_EVAL=off`, so the policy must use class-name matching
from the instruction and audio text. Earlier non-target-first runs were below
the 40% target and showed a strong center-button bias. The `v6` run is the
first preferred benchmark result above target: 47 successes out of 100
episodes.

Working interpretation:

- The model is not failing to open the gripper or generate trajectories in
  general; the target-first ablation and the final non-target-first `v6` both
  reach 47%.
- The main unsolved issue is binding the instruction's class words to the
  correct audio source and then binding that source's direction to the correct
  left/middle/right button trajectory.
- Random slot order is necessary to avoid the slot-0 shortcut, but fully random
  per-frame ordering can make the language prefix unstable across control
  steps. Target-agnostic azimuth canonicalization gives the sequence a stable
  physical interpretation without using the target class.
- The improvement from `v5` to `v6` suggests that the frozen/top-layer-unfreeze
  recipe was not enough for the VLM text stream to bind target class, source
  direction, and action. LoRA on `q_proj,v_proj` gave the model enough
  adaptation capacity while keeping the trainable parameter count lower than
  full LM unfreezing.
- Remaining weakness: `v6` still has weaker left performance than middle/right
  (9/36 on left), so the task is solved against the 40% criterion but not
  balanced.

### Final Preferred Two-Radio Run

The selected run is:

- Name: `smolvla_oracle_two_nl_azcanon_lora_40k_v6`
- Checkpoint dir:
  `outputs/smolvla_oracle_two_nl_azcanon_lora_40k_v6`
- Eval dir:
  `outputs/eval_smolvla_oracle_two_nl_azcanon_lora_40k_v6`
- Training steps: 40k
- Eval episodes: 100
- Audio fusion: `natural_language`
- Target-aware slot canonicalization: off
- SELD slot shuffle: on
- Target-agnostic azimuth canonicalization: on
- VLM adaptation: LoRA on `q_proj,v_proj`
- LoRA rank/alpha/dropout: `r=16`, `alpha=32`, `dropout=0.05`
- Regularization: image color jitter 0.05, image translate 2 px, state noise
  0.003, direction dropout 0.05
- Result: 0.47 success rate, 0.60 intention success rate

Final `v6` by-position result:

| Position | Success | Intention success |
| --- | ---: | ---: |
| left | 9/36 = 25.0% | 12/36 = 33.3% |
| middle | 21/33 = 63.6% | 26/33 = 78.8% |
| right | 17/31 = 54.8% | 22/31 = 71.0% |

To reproduce `v6` manually:

```bash
setsid bash -c 'env \
  DO_GENERATE=0 DO_CONVERT=0 DO_TRAIN=1 DO_EVAL=1 \
  EVAL_N=100 EVAL_SAVE_VIDEO=0 \
  AUDIO_FUSION_MODE=natural_language \
  TARGET_FIRST_SLOTS=off TARGET_FIRST_AUDIO_SLOTS_EVAL=off \
  SHUFFLE_SLOTS=on SHUFFLE_AUDIO_SLOTS_EVAL=on \
  CANONICALIZE_SLOTS=azimuth CANONICALIZE_AUDIO_SLOTS_EVAL=azimuth \
  TARGET_SLOT_PROTECT=on TRAIN_STEPS=40000 SAVE_EVERY=4000 LOG_EVERY=200 \
  OUTPUT_DIR=/home/rllab/Desktop/AVLABench/outputs/smolvla_oracle_two_nl_azcanon_lora_40k_v6 \
  EVAL_DIR=/home/rllab/Desktop/AVLABench/outputs/eval_smolvla_oracle_two_nl_azcanon_lora_40k_v6 \
  AUDIO_MAX_LEN=128 UNFREEZE_LM_LAYERS=0 \
  VLM_LORA=1 LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
  LORA_TARGET_MODULES=q_proj,v_proj \
  BATCH_SIZE=24 NUM_WORKERS=8 \
  IMAGE_COLOR_JITTER=0.05 IMAGE_TRANSLATE_PX=2 STATE_NOISE_STD=0.003 \
  DIRECTION_DROPOUT=0.05 AUDIO_CONF_DROPOUT=0.0 \
  bash sh/train_smolvla_oracle_two_radios.sh \
  > /tmp/smolvla_oracle_two_nl_azcanon_lora_40k_v6.direct.log 2>&1'
```

After completion, check:

```bash
jq . outputs/eval_smolvla_oracle_two_nl_azcanon_lora_40k_v6/eval_summary.json
```

For comparison, `v5` used the same target-agnostic azimuth canonicalization and
natural-language fusion but unfroze the top 6 LM layers instead of using LoRA.
It scored 0.29. The LoRA change is therefore part of the final recipe.

If a future rerun of `v6` regresses below 0.40, the next planned variant was:

| Variant | Output suffix | Changed knobs |
| --- | --- | --- |
| `v7` | `smolvla_oracle_two_class_tokens_azcanon_lm6_40k_v7` | `AUDIO_FUSION_MODE=class_tokens AUDIO_MAX_LEN=96 UNFREEZE_LM_LAYERS=6 CLASS_TOKEN_SCALE=0.2` |

Implementation ideas if stronger/better-balanced performance is needed:

- Add an explicit target-agnostic spatial summary token per canonicalized slot,
  e.g. `slot left/middle/right`, while still not using instruction target
  class. This is similar to azimuth words but may be easier for the action
  expert to consume.
- Add a small cross-attention or gating module from language target class
  embedding to audio slot embeddings inside the model. This would let the
  model learn class-to-source matching without reordering slots in
  preprocessing.
- Add an auxiliary training loss that predicts the target source azimuth from
  the fused prefix representation. The label can be derived from the dataset's
  target source during training, but eval would still use normal SmolVLA action
  generation.
- Reduce per-step eval slot instability by keeping an episode-level canonical
  source assignment based on azimuth and class history, rather than sorting each
  frame independently. This remains SELD post-processing as long as it does not
  use the instruction target.
- Increase visual/audio balance by raising `AUDIO_LR`, lowering LM unfreeze, or
  adding dropout to image features. The existing failures suggest the model may
  be overusing visual/default center priors.

## Useful Paths

- Best checkpoint seen so far:
  `outputs/smolvla_oracle_ver5/ckpt_step0040000.pt`
- Best summary:
  `outputs/eval_smolvla_oracle_ver5/eval_summary.json`
- Latest checked summary:
  `outputs/eval_smolvla_oracle_ver7/eval_summary.json`
- Latest checked two-radio summary:
  `outputs/eval_smolvla_oracle_two_nl_azcanon_lora_40k_v6/eval_summary.json`
- Auto-loop helper:
  `scripts/auto_smolvla_oracle_loop.sh`
- Two-radio loop helper:
  `scripts/auto_smolvla_two_radio_nl_loop.sh`

## Operational Notes

- No two-radio training or eval process was running when the `v6` result was
  recorded.
- Some runs appear to have been launched from a Claude Code shell session, so
  output ownership should be treated as shared workspace state rather than
  attributed to a single agent.
