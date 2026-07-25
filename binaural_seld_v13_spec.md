# Binaural Semantic Acoustic Imaging — Implementation Spec

Build a binaural (2-channel) sound event localization system that outputs, per 10 Hz frame,
a set of `(event_class, score, spherical_mask)` detections on a 180x360 elevation-azimuth grid.

Adapted from "Audio-Only Semantic Acoustic Imaging with Recognition-Prior Score Fusion"
(DCASE2026 Task 3, Wang et al.), retargeted from a 4-mic compact array to binaural.

**Read Sections 1 and 2 fully before writing any code.** They exist to prevent
whole classes of wasted work.

---

## 1. Anti-patterns — violating these silently produces a broken model

These are ranked. #1 is the one that will actually kill the project.

| # | Do NOT | Why |
|---|--------|-----|
| 1 | Resample the detector path below 32 kHz | Elevation and front/back cues live in pinna spectral notches at 5-16 kHz. At 16 kHz the Nyquist is 8 kHz and the capability is **gone**. Use 48 kHz. |
| 2 | Use mel / log-mel for the detector front-end | Mel compresses 5-20 kHz — exactly the elevation band. Mel also averages adjacent bins, which destroys phase. Linear frequency only. |
| 3 | Train on a single HRTF subject | The model memorizes one head and transfers to nothing. Use >= 8 subjects, hold out >= 2 for test. |
| 4 | Mono-downmix (L+R)/2 for the AudioMAE branch | Head-sized ITD (~0.66 ms) makes a comb notch at ~760 Hz, in the middle of the formant range. Run the encoder on L and R separately, concat embeddings. |
| 5 | Skip Stage-1 (class-agnostic synthetic pretraining) | 89% of the original compute is there. Stage 2 alone from random init will not learn the spatial mapping. |
| 6 | Add GCC-PHAT | It is an exact bijection of the `cos/sin dphi` channels (IFFT is invertible) — zero new information. It also collapses the frequency axis, which is the token axis of the cross-attention. And with one mic pair it only yields ITD (the cone of confusion), contributing nothing to elevation. |
| 7 | Put lag-domain features on the F axis | Lag bins and frequency bins mean different things. If you must add lag features, add them as separate K/V tokens to the cross-attention. |
| 8 | Reuse the source paper's 4-yaw-view augmentation | It exploits array rotational symmetry. A head is not rotationally symmetric — rotating the scene changes the HRTF. You must re-render. |
| 9 | Omit the frequency positional encoding on K/V | Convolutions are translation-equivariant along F, so the feature at bin f does not encode f. Without a positional encoding the direction queries cannot be frequency-selective, and the whole spectral-cue mechanism fails. **This is the easiest bug to miss.** |
| 10 | Report a single averaged localization metric | Azimuth accuracy dominates the average and hides total failure on the median plane. Evaluate the median-plane subset separately (Section 8.3). |

---

## 2. Why the architecture deviates from the source paper

Three changes, all downstream of one fact: **in a compact array, direction lives entirely in
inter-channel phase; in binaural, elevation and front/back live in the magnitude spectrum,
entangled with the source spectrum.**

Observed magnitude is `|X_L(f)| = |S(f)| * |H_L(f, az, el)|` — a product. Separating the
HRTF from the source spectrum is ill-posed without knowing what the source is.

| Change | From | To | Reason |
|---|---|---|---|
| Front-end channels | 12 (4 mic x 3) | 10 (Section 4.1) | Only the interaural *difference* carries direction; IPD is bilinear so give it explicitly |
| Prior fusion | Score re-ranking only | FiLM conditioning **+** score re-ranking | On the median plane ILD ~ 0 and dphi ~ 0; elevation is only in the monaural spectrum, so `where` depends on `what`. Late fusion is structurally too late. |
| Query dim `D_Q` | 16 | 32 | Each query must encode an HRTF spectral template, not a smooth steering vector. HRTF magnitude PCA typically needs 10-25 components; the attention logit matrix is rank-capped by `D_Q`. |

Unchanged and correct as-is: the 45x90 spherical grid, the mask decoder, the linear frequency
axis, the polar (log-mag, sin/cos phase) parameterization, the two-stage curriculum.

The mask output is a genuinely good fit for binaural: azimuth precision is a few degrees while
elevation is ~10-20 deg and front/back confusions are real. A mask can honestly render an
elongated blob or a bimodal front/back region. A point-DOA output cannot.

---

## 3. Repo layout

```
bsai/
  config.py            # single dataclass, all hyperparameters, no magic numbers elsewhere
  frontend.py          # waveform -> 10-channel TF tensor
  backbone.py          # ConvNeXt encoder
  spherical.py         # learned grid queries + cross-attention + temporal aggregation
  decoder.py           # Mask2Former-style mask decoder
  prior.py             # frozen audio encoder + MLP head + FiLM generator
  model.py             # assembles the above
  losses.py            # Hungarian matching + all loss terms
  data/
    brir.py            # BRIR bank generation (offline, CPU)
    scenes.py          # scene composition + mask rasterization
    dataset.py         # torch Dataset / DataLoader
  train.py             # stage 1 / stage 2 / prior
  evaluate.py          # mAP, AP50, Pearson r, median-plane subset
  export.py            # 10 Hz JSON + point compression
scripts/
  make_brirs.py        # run once, multiprocess
  compute_norm_stats.py
  bench.py             # step time + VRAM, run before committing to a long job
tests/
  test_frontend.py     # THE important one, see Section 9 Gate 1
  test_shapes.py
```

Config lives in exactly one place. Every number in this document goes into `config.py`.

---

## 4. Module specs

### 4.0 Shape reference — single source of truth

For `B=1`, one 2.0 s segment at 48 kHz. All shapes are exact; assert them.

```
waveform                     (B, 2, 96000)
padded waveform              (B, 2, 96480)          # +480 so n_frames is even
STFT complex                 (B, 2, 1024, 202)      # 1025 bins, Nyquist dropped
frontend out                 (B, 10, 202, 1024)     # (B, C, T, F) for Conv2d
convnext stage1              (B, 96,  202, 256)
convnext stage2              (B, 192, 101, 128)
convnext stage3              (B, 384, 101, 64)
convnext stage4              (B, 768, 101, 32)
grid queries                 (45, 90, 32)           # learned parameter
spherical per stage          (B, 101, 45, 90, 32)
spherical summed             (B, 101, 45, 90, 32)
F_sph (after temporal agg)   (B, 21, 32, 45, 90)
mask_feat_lo                 (B*21, 256, 45, 90)
mask_feat_hi                 (B*21, 32, 180, 360)
decoder queries              (B*21, 16, 256)
mask logits (final)          (B*21, 16, 180, 360)
class logits                 (B*21, 16, 14)         # 13 classes + no-object
score logits                 (B*21, 16, 1)
```

Constants: `T_STFT=202`, `F_BINS=1024`, `T_DEC=21`, `D_Q=32`, `H_S=45`, `W_S=90`,
`N_QUERIES=16`, `N_DEC_LAYERS=10`, `N_CLASSES=13`, `SR=48000`.

Why `202`: 2.0 s at 10 ms hop = 200, +2 from padding. It must be even so stage1->stage2
halves cleanly to 101. Why `21`: 2.0 s at 10 Hz = 20 output frames, +1 endpoint. The chain
is 100 Hz -> 50 Hz -> 10 Hz, and `T_DEC` matching the output rate means decoder frames map
1:1 to DCASE frames with no resampling.

### 4.1 Front-end (`frontend.py`)

STFT: `n_fft=2048`, `hop_length=480`, `win_length=2048`, `hann`, `center=True`.
At 48 kHz this is a 42.7 ms window, 10 ms hop, 23.4375 Hz per bin.

```python
DF = SR / N_FFT           # 23.4375 Hz
TAU_MAX = 0.8e-3          # s; head ITD max ~0.66 ms, headroom
EPS = 1e-5

wav = F.pad(wav, (0, 480))                                  # (B,2,96480)
X = torch.stft(wav.reshape(B*2, -1), n_fft=2048, hop_length=480,
               win_length=2048, window=hann, center=True, return_complex=True)
X = X[:, :1024, :].reshape(B, 2, 1024, 202)                 # drop Nyquist bin
XL, XR = X[:, 0], X[:, 1]                                   # (B,1024,202) complex

logm_L = torch.log(XL.abs() + EPS)
logm_R = torch.log(XR.abs() + EPS)
phi_L, phi_R = torch.angle(XL), torch.angle(XR)

Xc     = XL * XR.conj()                                     # cross-spectrum
dphi   = torch.angle(Xc)                                    # = wrap(phi_L - phi_R), no wrap bug
ild    = logm_L - logm_R

# local group delay: unambiguous at ALL frequencies (see note below)
Xc_pair = Xc[:, 1:, :] * Xc[:, :-1, :].conj()
gd = torch.angle(Xc_pair) / (2 * math.pi * DF)              # seconds, (B,1023,202)
gd = F.pad(gd, (0, 0, 1, 0), mode='replicate')              # -> (B,1024,202)
gd = (gd / TAU_MAX).clamp(-1.0, 1.0)

ch = torch.stack([logm_L, logm_R,
                  phi_L.cos(), phi_L.sin(), phi_R.cos(), phi_R.sin(),
                  dphi.cos(), dphi.sin(),
                  ild, gd], dim=1)                          # (B,10,1024,202)
ch = ch.transpose(2, 3)                                     # (B,10,202,1024)
```

**Why `gd` and not frequency-normalized IPD (`dphi / 2*pi*f`).** NIPD is only valid where the
phase has not wrapped: `f < 1/(2*tau)`. With `tau = 0.66 ms` that is `f < 758 Hz` = the first
32 of 1024 bins, ~3% of the axis — and in that narrow band `cos(dphi)` is already a gentle
half-period ramp that a conv reads easily. NIPD earns almost nothing.

The local group delay is different. `angle(Xc(f+1) * conj(Xc(f))) = 2*pi*DF*tau`, and
`2*pi*23.44*0.00066 = 0.097 rad << pi`, so it **never wraps, at any frequency**. A single
source at delay `tau0` becomes a flat line at height `tau0` across the entire spectrum instead
of a 16-period oscillation. That is the useful reparameterization. It is noisy (it is a
derivative) — let the conv smooth it.

**Why absolute phase is kept** despite only the difference carrying direction: a systematic
binaural-SSL feature ablation (arXiv 2511.13487) found that `Phase L/R + ILD + IPD` gives the
best **out-of-domain** generalization, especially under HRTF mismatch — which is exactly the
condition here. `ILD` is a linear function of `logm_L/logm_R` and in principle free, but the
same study found explicit provision still helps. Representation capacity != optimization.

**Normalization.** Run `scripts/compute_norm_stats.py` over 512 random training samples to get
per-channel mean/std over `(T, F)`, save to `norm_stats.pt`, register as buffers, standardize
at the front-end output. Do NOT hand-pick constants: `logm` spans ~15 in natural log while
`cos/sin` are bounded in [-1,1], and that scale mismatch is a real training-stability hazard.
Additionally subtract the per-sample mean of `(logm_L + logm_R)/2` before standardizing, to
make the model level-invariant. This preserves `ild` (a common mean cancels in a difference).

Config flags for ablation: `use_gd`, `use_abs_phase`, `use_ild`, `use_nipd` (default off).
Changing these changes `in_channels`; derive it from the flags, do not hardcode 10 twice.

### 4.2 Backbone (`backbone.py`)

ConvNeXt-T geometry: `depths=(3,3,9,3)`, `dims=(96,192,384,768)`, ~28M params.
Block: `DWConv7x7 -> LN -> Linear(d,4d) -> GELU -> Linear(4d,d) -> LayerScale(1e-6) -> DropPath -> residual`.
`drop_path_rate=0.1` linearly ramped across blocks.

Downsampling is **asymmetric** — frequency is reduced 5x more than time:

| Layer | Op | T | F | C |
|---|---|---|---|---|
| stem | `Conv2d(10, 96, k=(3,4), s=(1,4), p=(1,0))` + LN | 202 | 1024->256 | 96 |
| stage 1 | 3 blocks | 202 | 256 | 96 |
| down 1 | LN + `Conv2d(96, 192, k=2, s=2)` | 202->101 | 256->128 | 192 |
| stage 2 | 3 blocks | 101 | 128 | 192 |
| down 2 | LN + `Conv2d(192, 384, k=(1,2), s=(1,2))` | 101 | 128->64 | 384 |
| stage 3 | 9 blocks | 101 | 64 | 384 |
| down 3 | LN + `Conv2d(384, 768, k=(1,2), s=(1,2))` | 101 | 64->32 | 768 |
| stage 4 | 3 blocks | 101 | 32 | 768 |

The stem uses `k=(3,4)` (not `(1,4)`) so the very first layer sees temporal context; stride
stays `(1,4)` to keep `T=202`.

Return stage 2, 3, 4 outputs. Stage 1 is not projected to the sphere.

Note: unlike vision ConvNeXt, the time axis stays at 101 through stages 2-4, so the deep stages
are ~66x more tokens than ImageNet ConvNeXt's 7x7. Stage 3 is the single most expensive block.
**Enable gradient checkpointing on stages 3 and 4.**

For binaural specifically, the elevation cue is fine high-frequency magnitude structure, and by
stage 4 there are only 32 bins (750 Hz/bin) — too coarse to track a pinna notch. Stage 2
(128 bins, 187 Hz/bin) carries almost all of it. The three stages are summed with equal weight
by default; add `sph_stage_weights: tuple[float,float,float] = (1,1,1)` to config as a
learnable-or-fixed ablation knob.

### 4.3 Spherical cross-attention (`spherical.py`)

```python
# parameters
grid_q = nn.Parameter(torch.randn(45*90, D_Q) * 0.02)          # 4050 x 32
freq_pe = nn.ParameterDict({                                    # CRITICAL — see anti-pattern #9
    's2': nn.Parameter(torch.randn(128, D_Q) * 0.02),
    's3': nn.Parameter(torch.randn(64,  D_Q) * 0.02),
    's4': nn.Parameter(torch.randn(32,  D_Q) * 0.02),
})
k_proj = {s: nn.Linear(C_s, D_Q) for s in ('s2','s3','s4')}
v_proj = {s: nn.Linear(C_s, D_Q) for s in ('s2','s3','s4')}
N_HEADS = 4     # head_dim = 8

# forward, per stage
feat = feat.permute(0, 2, 3, 1)                # (B, 101, F_s, C_s)
K = k_proj[s](feat) + freq_pe[s]               # (B, 101, F_s, D_Q)
V = v_proj[s](feat)
Q = grid_q.unsqueeze(0) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # FiLM, (B, 4050, D_Q)
Q = Q.unsqueeze(1).expand(B, 101, 4050, D_Q)

# fold time into batch, then multi-head
q = Q.reshape(B*101, 4050, N_HEADS, D_Q//N_HEADS).transpose(1, 2)
k = K.reshape(B*101, F_s,  N_HEADS, D_Q//N_HEADS).transpose(1, 2)
v = V.reshape(B*101, F_s,  N_HEADS, D_Q//N_HEADS).transpose(1, 2)
o = F.scaled_dot_product_attention(q, k, v)                    # (B*101, H, 4050, dh)
sph_s = o.transpose(1, 2).reshape(B, 101, 45, 90, D_Q)
```

Attention is **per time frame** (T folded into batch). That is what makes this a
frequency -> direction mapping: at instant t, each of the 4050 direction queries pools evidence
across that frame's frequency tokens. Do not attend across time here.

Use `scaled_dot_product_attention` — do not materialize the `(101, 4050, 128)` weight matrix.

Sum the three stages at 101-frame resolution. This is the whole point of projecting to the
sphere first: `(101,128,192)`, `(101,64,384)`, `(101,32,768)` cannot be added, but after
projection all three are `(101,45,90,32)`. The sphere is the common coordinate system.
Sum, do not concat.

Temporal aggregation 101 -> 21:

```python
x = sph.permute(0, 2, 3, 4, 1).reshape(B*45*90, D_Q, 101)
x = F.pad(x, (2, 2))                                   # -> 105
x = self.temporal_conv(x)                              # Conv1d(D_Q, D_Q, k=5, s=5) -> 21
F_sph = x.reshape(B, 45, 90, D_Q, 21).permute(0, 4, 3, 1, 2)   # (B,21,32,45,90)
```

`105 = 21*5` exactly. Verify with an assert.

**Known limitation, do not try to fix in v1:** the 45x90 equirectangular grid oversamples the
poles (at el=88 deg, 90 azimuth cells split a tiny circle) and undersamples the equator. Because
each cell owns an *independent* learned query, the grid can absorb the distortion. Note it and
move on.

### 4.4 Mask decoder (`decoder.py`)

Standard Mask2Former. `d_model=256`, 16 queries, 10 layers, 8 heads, FFN dim 2048.
Time is folded into batch: the decoder runs per output frame, `B*21` independent problems.

```
query_feat = nn.Embedding(16, 256)        # content
query_pos  = nn.Embedding(16, 256)        # positional
grid_pos   = nn.Parameter(45*90, 256)     # learned 2D positional encoding for the sphere

mask_feat_lo = Conv2d(D_Q, 256, 1)(F_sph)                       # (B*21, 256, 45, 90)
mask_feat_hi = up(F_sph): D_Q -> 64 (ConvT k2 s2) -> 32 (ConvT k2 s2)   # (B*21, 32, 180, 360)
```

Per layer `l` (this exact order — it is Mask2Former's, and it matters):
1. **masked** cross-attention: `Q=queries+query_pos`, `K/V=mask_feat_lo+grid_pos`,
   `attn_mask = (sigmoid(mask_{l-1}) < 0.5)`. If a query's mask is empty everywhere, unmask
   that row entirely, otherwise you get NaNs.
2. self-attention over the 16 queries
3. FFN

Heads (applied after every layer for deep supervision, and at the end for output):
```
class_head    : Linear(256, 14)
score_head    : Linear(256, 1)
mask_embed_lo : MLP(256 -> 256 -> 256)   ->  mask_lo = einsum('bqc,bchw->bqhw', e, mask_feat_lo)
mask_embed_hi : MLP(256 -> 256 -> 32)    ->  mask_hi = einsum('bqc,bchw->bqhw', e, mask_feat_hi)
```
`mask_lo` (45x90) drives the next layer's attention mask and deep supervision.
`mask_hi` (180x360) is computed **only from the final layer** — it is 21.8M values per sample;
computing it per layer will OOM.

`mask_feat_hi` uses 32 channels, not 256. At 256 it would be 348M values per sample.

### 4.5 Prior + FiLM (`prior.py`)

```python
class PriorEncoder(Protocol):
    out_dim: int
    def __call__(self, wav_mono_16k: Tensor) -> Tensor: ...   # (B, out_dim)
```

Default: AudioMAE (frozen), 128-mel fbank @ 16 kHz, zero-pad time to the checkpoint's
pretrained frame count, mean-pool the encoder output. Keep it behind the Protocol so it can be
swapped (BEATs, PANNs CNN14) without touching the model — the checkpoint may be awkward to
obtain and this is not the place to get blocked.

```python
emb = cat([enc(resample(wav_L, 16k)), enc(resample(wav_R, 16k))], -1)   # (B, 2*out_dim)
p_class = sigmoid(MLP(2*out_dim -> 512 -> 13)(emb))                     # (B, 13)
gamma, beta = MLP(2*out_dim -> 256 -> 2*D_Q)(emb).chunk(2, -1)          # (B, D_Q) each
```

L and R go through **separately**, embeddings concatenated. Never `(L+R)/2` (anti-pattern #4).

FiLM injects `(gamma, beta)` into the spherical grid queries (Section 4.3). This is the change
that matters: on the median plane the only elevation evidence is the monaural spectrum,
entangled with the source spectrum, so the direction queries need to know what the source is
before they can read the notch pattern. Score-level fusion alone cannot supply that.

The encoder is frozen, so **cache embeddings to disk keyed by scene id**. This makes the prior
head's own training take minutes instead of hours and removes the encoder from the detector's
training step entirely.

Initialize the FiLM head's final layer to zeros so `gamma=beta=0` at step 0 — training starts
from the unconditioned model and learns to use the prior.

### 4.6 Fusion (`model.py`)

Align the prior to the output timeline: split the recording into non-overlapping 2 s windows
(= 20 frames at 10 Hz), assign each window's `p_class` to its frames as `p(c, t)`, clip the
last window at the boundary.

```
s_f = s_d * p(c, t) ** alpha        # alpha = 0.5
```
Re-ranking only. The predicted class and mask are kept from the detector. Both FiLM and this
score fusion are active; they are not alternatives.

---

## 5. Losses (`losses.py`)

Hungarian matching per `(batch, frame)`, cost on the **low-res** 45x90 mask (4050 points is
already cheap — skip Mask2Former's point sampling):

```
C = 2.0 * (-p_class[c_gt]) + 5.0 * BCE(mask_lo, gt_lo) + 5.0 * Dice(mask_lo, gt_lo)
```

Total loss:

| Term | Weight | Applied to |
|---|---|---|
| class CE (no-object weight 0.1) | 2.0 | all 16 queries |
| mask BCE | 5.0 | matched, `mask_hi` @ 180x360 |
| mask Dice | 5.0 | matched, `mask_hi` @ 180x360 |
| score BCE (target = is-matched) | 1.0 | all 16 queries |
| deep supervision (`mask_lo` BCE+Dice+CE) | same weights | layers 1..9 |

Full 180x360 BCE is 21.8M elements per sample — fine as elementwise ops, no point sampling
needed.

**Stage 1 is class-agnostic**: the class head is 2-way (object / no-object) and the CE term
degenerates to objectness. At Stage 2, **re-initialize the class head to 14-way** and load
everything else from the Stage-1 checkpoint. Make this explicit in `train.py`; a silent shape
mismatch here will be loaded as "missing key" and ignored.

---

## 6. Data pipeline

No binaural version of STAIRS26 exists. Converting the 32-ch Eigenmike recordings to binaural
via Ambisonics is possible but **destroys the cue you care about**: order-4 HOA from a 4.2 cm
array is spatially accurate only to ~5 kHz, and the elevation notches live above that. Any
elevation cue you recovered that way would be a fabrication of the decoder.

Therefore: **fully synthetic, rendered with measured HRTFs.** This is what BiSELD did
(arXiv 2507.20530, "Binaural Set"). Acquire a small real binaural test set separately if you
can; treat that as out of scope for v1.

### 6.1 BRIR bank (`scripts/make_brirs.py`, run once)

The naive approach — render 500k complete scenes — is CPU-bound and dominates wall time by
10-60x over the GPU step. Render **BRIRs**, not audio, and compose scenes on the fly.

```python
import pyroomacoustics as pra
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D

hrtf = MeasuredDirectivityFile(path=sofa_path, fs=48000, interp_order=12, interp_n_points=1000)
orientation = Rotation3D([yaw, pitch], "yz", degrees=True)
dir_l = hrtf.get_mic_directivity("left",  orientation=orientation)
dir_r = hrtf.get_mic_directivity("right", orientation=orientation)
# the two mics must be CO-LOCATED at the head position
room.add_microphone_array(pra.MicrophoneArray(np.c_[head_pos, head_pos], fs, directivity=[dir_l, dir_r]))
```

pyroomacoustics ships a bundled MIT KEMAR SOFA file; use it for a smoke test, but for real
training use a multi-subject database (CIPIC, HUTUBS, ARI, SADIE II).

Per room:
- dims `x,y ~ U(2,10)`, `z ~ U(2,4)`; absorption `~ U(0.2, 0.7)`; `max_order=5`
- head at a random position >= 0.5 m from walls, random yaw
- one randomly chosen HRTF subject **per room** (a room has one head)
- **20 cluster centers**, each with **4 sub-positions** within a 5-15 deg angular offset
  = 80 source positions per room

The clustering matters: an extended source is rendered as multiple sub-sources inside an
angular support, so the bank must contain sets of *nearby* directions. Randomly scattered
positions cannot produce extended sources.

Store per BRIR: `brir (2, 24000)` fp32 (0.5 s), `az`, `el`, `distance`, `room_id`,
`cluster_id`, `hrtf_subject`.

Targets: **600 rooms x 80 positions = 48,000 BRIRs, ~9 GB.**

Split by **HRTF subject and room**, not by sample:
- train: subjects 1-7, 90% of their rooms
- val: subjects 1-7, 10% of their rooms  (in-domain)
- test: subjects 8-10, all rooms          (**HRTF-mismatched — this is the number that matters**)

Multiprocess over rooms. If throughput is unacceptable, drop `max_order` to 3
(343 images vs 1331, ~4x faster) before reducing the bank size.

### 6.2 Scene composition (`data/scenes.py`, on the fly)

```
pick a room
n_src ~ U{1..6} clusters from that room
for each cluster:
    clip = random 2 s source clip
    sig  = sum over sub-sources of conv(clip, brir_subsource)     # FFT conv
    mix += sig * random_gain
target = per-cluster mask
```

Decoupling BRIRs from source clips is what makes this cheap: 48k BRIRs x an unbounded clip
pool gives effectively unlimited scene variety at ~zero CPU cost per sample.

Mask rasterization: for each sub-source direction, splat a disk of radius 3 deg on the
180x360 grid; the union over one cluster's sub-sources is that cluster's target region.
Downsample (max-pool) to 90x180 and 45x90 for the low-res targets.

Augmentation:
- **L/R mirror** (swap channels, `az -> -az`, mirror the mask): 2x, valid under head symmetry
- gain, SNR, background noise, clip choice
- **NOT** the 4-yaw-view trick (anti-pattern #8)

Source clips:
- Stage 1: VCTK, class-agnostic
- Stage 2: 13 STARSS/STAIRS classes — female speech, male speech, clapping, telephone,
  laughter, domestic sounds, walk/footsteps, door open/close, music, musical instrument,
  water tap/faucet, bell, knock. FSD50K + VCTK covers these; put the class->clip-list mapping
  in a JSON config, not in code.
  Target >= 500 clips per class, split by clip so no clip crosses train/test.

### 6.3 Data volume summary

| Item | Amount | Disk |
|---|---|---|
| BRIR bank | 48,000 | ~9 GB |
| VCTK (stage 1 clips) | ~44 h | ~10 GB |
| Event clip pool (stage 2) | >= 6,500 clips | ~10-30 GB |
| Cached prior embeddings | per scene id | ~1-3 GB |
| Stage-1 samples seen | 500,000 | composed on the fly |
| Stage-2 samples seen | 200,000 | composed on the fly |

---

## 7. Training (`train.py`)

The source paper used batch size 1 — a memory constraint of its setup, not a design choice.
Estimated activation memory here is ~2-3 GB/sample, so a 32 GB card should take batch 8 with
bf16 + gradient checkpointing. Batch 1 starves the GPU; the dominant cost is a
bandwidth-bound depthwise-conv workload whose MFU at batch 1 is in the single digits.
**Measure with `scripts/bench.py` before committing.** Fall back to batch 4 + grad accum 2.

| Stage | Samples | Batch | Steps | LR | Schedule |
|---|---|---|---|---|---|
| 1 — detector, class-agnostic | 500k | 8 | 62.5k | 3e-4 | AdamW wd 0.05, cosine, 5k warmup |
| 2 — detector, 13-class | 200k | 8 | 25k | 1e-4 | AdamW wd 0.05, cosine, 1k warmup |
| prior — MLP head only | 240k | 256 | 10k | 1e-3 | AdamW, embeddings cached |

LRs are the paper's (1e-4 / 3e-5 at batch 1) scaled by ~sqrt(8). bf16 AMP throughout;
grad clip 1.0. Save checkpoints every 2k steps and keep the last 3.

Stage 2 loads Stage 1 weights except the class head (Section 5). FiLM is active in both stages;
in Stage 1 the prior head is untrained, so either freeze FiLM to zeros in Stage 1 or pretrain
the prior head first. **Recommended order: prior head -> stage 1 (FiLM live) -> stage 2.**

---

## 8. Resource budget

Estimates, not measurements. Gate 6 replaces them with real numbers.

### 8.1 Compute

Per training step (fwd+bwd) ~1 TFLOP at batch 1, dominated by ConvNeXt stage 3. An RTX 5090 is
~210 TFLOPS bf16 dense, but expected MFU is 5-15% (depthwise convs are bandwidth-bound; 4050
queries at head_dim 8 use tensor cores poorly).

| Item | Estimate |
|---|---|
| BRIR rendering (one-time, 16 CPU cores) | 6-20 h |
| Stage 1 — 62.5k steps @ batch 8 | 4-8 h |
| Stage 2 — 25k steps @ batch 8 | 2-3 h |
| Prior head (embeddings cached) | < 10 min |
| **GPU total** | **6-11 h** |
| Peak VRAM @ batch 8, bf16 + checkpointing | ~18-26 GB |

The real bottleneck is CPU-side BRIR rendering, not the GPU. Budget for it.

### 8.2 RTX 5090 specifics

Blackwell is `sm_120`: **CUDA 12.8+ and PyTorch 2.7+ are hard requirements.** An older wheel
fails with a no-kernel-image error. Pin them in `requirements.txt` and assert
`torch.cuda.get_device_capability() >= (12, 0)` at startup.

### 8.3 Evaluation (`evaluate.py`)

Metrics: mAP (primary), AP50, Pearson r (correlation of rendered energy maps for matched pairs).
The reference system scored 0.1017 / 0.2378 / 0.7904 on a 4-mic array. Expect **worse** — you
have half the channels and a harder cue structure. The Pearson-r-high / mAP-low gap in that
system indicated candidate ranking, not mask evidence, was the bottleneck; watch whether the
same holds here.

Report these subsets **separately** — an average will hide total median-plane failure:

| Subset | Definition |
|---|---|
| horizontal | \|el\| < 10 deg |
| **median plane** | \|az\| < 15 deg or \|az - 180\| < 15 deg, el varying |
| front/back | report confusion rate: predictions mirrored about the coronal plane |
| **HRTF-mismatched** | test subjects 8-10 (the headline generalization number) |

Export (`export.py`): 10 Hz, score threshold 0.05, max 6 detections/frame, mask-energy
threshold 0.10, serialized as `(x=az_idx, y=el_idx, e=intensity)` points on the 180x360 grid.

Point compression, if file size becomes a problem: the evaluator renders submitted points with
a sigma=6 deg spherical Gaussian and thresholds at 10% of peak, so only points that survive that
rendering matter. Drop points below 10% of the mask peak, keep the strongest point per local
grid cell (2 px cells for score >= 0.20, 6 px cells + 2 px boundary support otherwise). This cut
the reference system's max JSON from 149 MB to 19 MB with mAP moving 0.1017 -> 0.1009. Defer
until needed.

---

## 9. Build order and verification gates

Do not skip gates. Each one catches a class of bug that is far more expensive to find later.
Report the gate output, then continue.

**Gate 0 — env.** `torch>=2.7`, CUDA 12.8+, capability check, `pyroomacoustics>=0.8`, `sofar`,
`scipy`, `soundfile`. Assert the GPU is visible.

**Gate 1 — front-end physics test.** `tests/test_frontend.py`. This is the highest-value test
in the project; it validates all the phase math in ~15 lines.
```
Synthesize white noise x. Build wav = stack([x, delayed(x, tau=0.5ms)]).
Run the front-end.
ASSERT: gd channel (denormalized by TAU_MAX) ~= 0.5ms, within 5%, averaged over bins 50..900
        where the noise has energy.
ASSERT: dphi at bin f ~= wrap(2*pi*f*tau), within 0.05 rad, for f < 700 Hz.
ASSERT: ild ~= 0 within 0.01 (same signal, only delayed).
Then set wav = stack([x, 0.5*x]) (level difference, no delay):
ASSERT: ild ~= log(2) within 0.01. ASSERT: gd ~= 0.
```
If `gd` comes back as `-tau` you have the conjugate backwards. If it is `tau/2` your `DF` is wrong.

**Gate 2 — shapes.** `tests/test_shapes.py` asserts every row of the Section 4.0 table with a
random input. Cheap, and it makes the rest of the build a no-op debugging-wise.

**Gate 3 — spherical module.** Assert `F_sph` is `(1,21,32,45,90)`. Assert
`freq_pe` is actually in the graph (`grad is not None` after a backward). Assert the temporal
conv gives exactly 21 from 105.

**Gate 4 — overfit one sample.** Full model, 1 fixed sample, 500 steps, lr 1e-3, no
augmentation. Loss must reach < 0.05. Dump the predicted vs GT mask as a PNG.
**If this does not overfit, nothing downstream will work.** Debug here, not at step 40k.

**Gate 5 — overfit 100 samples.** 2000 steps. Loss should fall well below the 1-sample
plateau of a broken model. Confirms the data pipeline is not shuffling targets.

**Gate 6 — benchmark.** `scripts/bench.py`: step time and peak VRAM at batch 1/2/4/8. Report a
table. Pick the batch size from this, and recompute the Section 8.1 estimate with the real
number before launching a multi-hour job.

**Gate 7 — short stage-1 run.** 2k steps on the real pipeline. Loss must decrease monotonically
in trend. Check the data loader is not the bottleneck (GPU util > 80%).

**Gate 8 — full run.** Stage 1 -> Stage 2. Evaluate the Section 8.3 subsets at each checkpoint.

---

## 10. Open questions — decide these before Gate 4, do not guess silently

1. **HRTF database.** Which one, how many subjects, licensing. Everything downstream depends on
   this and it is the single biggest generalization lever.
2. **Prior encoder checkpoint.** AudioMAE weights are not on PyPI. If obtaining them blocks you,
   swap in another frozen encoder via the Protocol and note the substitution.
3. **Real binaural test set.** The whole pipeline is synthetic. Without any real recording the
   sim-to-real gap is unmeasured. Even 10 minutes of annotated real binaural audio would be
   informative.
4. **Decoder temporal context.** This spec runs the decoder per output frame (T folded into
   batch), matching the source paper's per-frame detection description. Cross-frame attention
   would likely help tracking. Out of scope for v1; do not add it unprompted.
5. **`D_Q = 32` is a guess** derived from HRTF PCA component counts. Ablate 16 / 32 / 64 once
   the pipeline runs.
