# Spatial Audio-Conditioned VLA Fine-tuning Pipeline

## Project Overview

이 프로젝트는 기존 Vision-Language-Action (VLA) 모델에 **공간음향 정보**를 결합하여, 로봇이 시각 정보뿐 아니라 소리의 클래스와 방위(azimuth, elevation)를 함께 이해하고 행동하도록 만드는 것을 목표로 한다.

### Core Goals

1. **Multimodal extension**: 기존 Vision + Language + Action 파이프라인에 Audio (spatial) modality 추가
2. **LLM common sense 활용**: Pretrained LLM의 상식 추론 능력을 audio 정보와 결합
   - SELD 결과가 애매할 때 visual evidence로 보정 (e.g., `siren 0.5` + visible alarm clock → alarm clock 선택)
   - 언어적 공간 표현 처리 (e.g., "소리나는 물체 **오른쪽**의 물체")
   - Common sense priors (e.g., "소리낼만한 물체" → radio, phone 우선순위)
3. **Efficient fine-tuning**: Pretrained VLA weight를 최대한 보존하면서 audio modality 주입
4. **Scalability**: SmolVLA로 검증 후 더 큰 VLA 모델로 확장 가능한 모듈 설계

### Base Model Choice: SmolVLA

- **Why**: 450M params, consumer GPU에서 학습 가능, fast iteration
- **Architecture**:
  - VLM backbone: SmolVLM2-500M (SigLIP vision encoder + SmolLM2 language decoder)
  - LLM: 원본 중 first 16 layers만 사용
  - Action expert: ~100M params flow-matching transformer
- **Default fine-tune recipe**: VLM freeze, action expert만 학습

---

## System Architecture

### High-Level Pipeline

```
Audio (multichannel) ──► SELD model ──► Top-K events {class, az, el, conf, time_offset}
                                              │
                                              ▼
                                   Audio Token Builder
                                   ├─ Path A: LLM-readable tokens
                                   └─ Path C: Direction map for vision grounding
                                              │
Image ──► SigLIP ──► Vision tokens ◄─ cross-attn ── (Path C)
                          │
Instruction ──► Tokenizer ─┤
                          │
State ──► Linear proj ────┤
                          ▼
                    SmolVLM2 Decoder
                [sys][inst][audio_tokens][vision_tokens][state]
                          │
                          ▼
                    Action Expert
                          │
                          ▼
                    Action Chunk
```

### Two-Path Audio Injection (Hybrid)

**Path A: Language-stream audio tokens** — LLM reasoning 활성화용
- Class name은 **기존 vocab의 텍스트 토큰**으로 (`siren`, `typing`)
- Direction은 **continuous learnable embedding** (sin/cos MLP)
- Top-K events를 모두 전달 (uncertainty 보존)

**Path C: Vision-audio cross-attention adapter** — Spatial grounding용
- Audio의 방향 정보가 vision feature의 해당 공간 위치에 주입
- Zero-init gated residual로 안전한 파인튜닝
- Camera FoV 밖의 소리는 Path A가 주로 처리

---

## Directory Structure

```
project_root/
├── README.md
├── configs/
│   ├── default.yaml              # 기본 하이퍼파라미터
│   ├── stage1_alignment.yaml     # Phase 0: audio module alignment
│   ├── stage2_main.yaml          # Phase 1: action expert + audio
│   └── stage3_lora.yaml          # Phase 2: LLM LoRA
├── data/
│   ├── raw/                       # 원본 수집 데이터 (rosbag, video, audio)
│   ├── processed/                 # LeRobot format
│   └── seld_cache/                # 미리 계산한 SELD 결과
├── src/
│   ├── audio/
│   │   ├── seld_wrapper.py       # SELD 모델 추론 래퍼
│   │   ├── audio_token_builder.py # Path A: audio → language tokens
│   │   ├── direction_encoder.py  # sin/cos MLP for (az, el)
│   │   └── augmentation.py       # SELD output augmentation
│   ├── models/
│   │   ├── smolvla_audio.py      # 확장된 SmolVLA policy
│   │   ├── vision_audio_adapter.py # Path C: cross-attn adapter
│   │   └── audio_projector.py    # embedding space alignment
│   ├── data/
│   │   ├── dataset.py            # AudioVLA dataset class
│   │   ├── collate.py            # batch collation with audio
│   │   └── transforms.py         # data augmentation
│   ├── training/
│   │   ├── train.py              # 메인 학습 스크립트
│   │   ├── trainer.py            # training loop
│   │   ├── schedulers.py         # param-group별 lr
│   │   └── callbacks.py          # checkpoint, eval, logging
│   ├── eval/
│   │   ├── evaluator.py          # 다양한 메트릭
│   │   ├── probing.py            # LLM knowledge probing
│   │   └── ablation.py           # ablation study runner
│   └── utils/
│       ├── coord_transform.py    # mic frame → robot frame
│       └── visualization.py
├── scripts/
│   ├── collect_data.py           # 데이터 수집 도구
│   ├── precompute_seld.py        # SELD 결과 사전 계산
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   └── run_eval.sh
└── notebooks/
    ├── exploration.ipynb
    └── result_analysis.ipynb
```

---

## Data Requirements

### Dataset Format

LeRobot dataset format을 기반으로 audio 정보 확장:

```
episode_N/
├── observation.images.main      # RGB frames (video)
├── observation.images.wrist     # (optional) wrist camera
├── observation.state            # robot joint state
├── action                       # action chunks
├── observation.audio.waveform   # multichannel audio (4ch ambisonic 권장)
├── observation.audio.seld       # 사전 계산된 SELD output (선택)
│   ├── class_probs: [T, num_classes]
│   ├── azimuth: [T, num_events]      # degrees
│   ├── elevation: [T, num_events]    # degrees
│   └── confidence: [T, num_events]
├── task                         # language instruction
└── meta.json                    # camera/mic extrinsic, sample rates 등
```

### Collection Requirements

- **마이크 어레이**: 최소 4ch ambisonic 또는 tetrahedral array
- **동기화**: Audio-video 시간 동기화 (< 10ms drift)
- **좌표계**: 마이크 → 카메라 → 로봇 base frame 변환 행렬 기록 필수
- **Sampling rate**: Audio 32kHz 또는 48kHz, video 30fps 이상

### Dataset Size Guidelines

| 단계 | 에피소드 수 | 목적 |
|---|---|---|
| POC | 50–100 | Audio 영향 확인 |
| 실험 | 300–500 | 유의미한 결과 |
| 논문 | 800–1500 | Ablation 포함 |

### Task Design Principles

"Audio가 의사결정에 영향을 미치는" 태스크 설계:

1. **Audio-dependent identification**: 시각적으로 비슷한 물체 중 소리로 구별
   - 예: 동일한 케이스의 라디오 2개 중 소리 나는 것
2. **Spatial language with audio**: Audio 방향 + 언어적 공간 표현
   - 예: "사이렌 소리 나는 물체의 오른쪽 물체를 집어"
3. **Common sense reasoning**: SELD 불완전 + visual evidence로 보정
   - 예: SELD가 애매할 때 장면에서 가장 가능성 높은 발음체 선택
4. **Negative samples**: 20–30%는 audio 없는 기본 태스크 (regression 방지)

---

## Implementation Details

### 1. SELD Wrapper (`src/audio/seld_wrapper.py`)

```python
class SELDWrapper:
    """
    SELD 모델 추론 래퍼.
    
    Input: multichannel audio waveform [C, T]
    Output: list of events per time step
        [{class_id, class_name, az, el, conf, time_offset}, ...]
    
    Window: 2-3초 sliding window, hop 250ms 권장
    """
    def __init__(self, model_name="seldnet_dcase2024", window_sec=2.5, hop_sec=0.25):
        ...
    
    def predict(self, audio_chunk):
        # Returns top-K events with confidence
        ...
    
    def get_recent_events(self, current_time, lookback_sec=3.0):
        # 현재 시점 기준 최근 이벤트들 반환
        ...
```

**설계 결정**:
- Window 2–3초: 1초는 transient 놓침, 5–10초는 reactivity 저하
- Top-K events 보존 (K=3–5): uncertainty 전파
- SELD는 사전 계산해서 cache 권장 (학습 중 overhead 제거)
- 좌표계는 이 wrapper 내부에서 robot base frame으로 변환

### 2. Audio Token Builder (`src/audio/audio_token_builder.py`)

```python
class AudioTokenBuilder:
    """
    SELD output을 LLM이 읽을 수 있는 토큰 시퀀스로 변환.
    
    출력 형식 (tokenize 전):
        "[AUDIO] siren at <DIR_EMB_1> confidence 0.6; 
                  also possibly alarm 0.3, phone 0.1.
                  typing at <DIR_EMB_2> confidence 0.7 [/AUDIO]"
    
    <DIR_EMB_i>는 direction embedding으로 대체되는 placeholder.
    """
    def __init__(self, tokenizer, direction_encoder, top_k=3):
        ...
    
    def build(self, events):
        # 1. 각 event를 텍스트 템플릿으로 변환 (class name은 vocab 토큰 사용)
        # 2. <DIR_EMB_i> placeholder를 위치에 삽입
        # 3. tokenize → direction embedding을 해당 위치에 injection
        ...
```

**중요 포인트**:
- Class명은 **반드시 기존 vocab**의 단어로 (pretrained LLM 지식 활용)
- Special token (`[AUDIO]`, `[/AUDIO]`)은 기존 vocab 유사 토큰 평균으로 초기화
- Confidence는 "confidence 0.6" 처럼 자연어 숫자로 (LLM이 uncertainty 이해)

### 3. Direction Encoder (`src/audio/direction_encoder.py`)

```python
class DirectionEncoder(nn.Module):
    """
    (azimuth, elevation) → LLM hidden dim embedding.
    """
    def __init__(self, llm_hidden_dim, internal_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, internal_dim),  # [sin_az, cos_az, sin_el, cos_el]
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, llm_hidden_dim),
        )
        # CRITICAL: small-scale init
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, az_deg, el_deg):
        az = torch.deg2rad(az_deg)
        el = torch.deg2rad(el_deg)
        features = torch.stack([
            torch.sin(az), torch.cos(az),
            torch.sin(el), torch.cos(el)
        ], dim=-1)
        return self.mlp(features)
```

**설계 결정**:
- sin/cos 인코딩: 각도의 순환성 (179° ≈ -179°) 자연스럽게 표현
- Small-scale output 초기화: 파인튜닝 초기에 LLM 입력을 교란하지 않음
- Learnable MLP: 학습 중 "embedding ↔ 언어적 공간 표현" alignment 형성

### 4. Vision-Audio Adapter (`src/models/vision_audio_adapter.py`)

```python
class AudioGroundingAdapter(nn.Module):
    """
    Path C: Vision token이 audio token을 query하는 cross-attention adapter.
    Zero-init gated residual로 초기엔 identity에 가까움.
    """
    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'ln_q': nn.LayerNorm(dim),
                'ln_kv': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
                'ln_ffn': nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        ])
        # Gate: 0으로 초기화 → 초기엔 residual만 통과 (identity)
        self.attn_gates = nn.Parameter(torch.zeros(num_layers))
        self.ffn_gates = nn.Parameter(torch.zeros(num_layers))
    
    def forward(self, vision_tokens, audio_tokens, audio_mask=None):
        x = vision_tokens
        for i, layer in enumerate(self.layers):
            q = layer['ln_q'](x)
            kv = layer['ln_kv'](audio_tokens)
            attn_out, _ = layer['cross_attn'](q, kv, kv, key_padding_mask=audio_mask)
            x = x + self.attn_gates[i].tanh() * attn_out
            
            ffn_out = layer['ffn'](layer['ln_ffn'](x))
            x = x + self.ffn_gates[i].tanh() * ffn_out
        return x
```

**핵심**: `tanh(gate)` with `gate=0` 초기화
- 초기: `tanh(0) = 0` → adapter가 출력에 영향 없음, pretrained 모델 그대로 동작
- 학습 진행: gate 값이 커지며 adapter 영향 점진적 증가
- 파인튜닝 안정성의 핵심 테크닉 (Flamingo, LLaMA-Adapter 등이 동일 원리)

### 5. SmolVLA Extension (`src/models/smolvla_audio.py`)

```python
class AudioAwareSmolVLAPolicy(SmolVLAPolicy):
    """
    SmolVLA를 audio modality로 확장.
    기존 forward path에 audio token 주입 + vision-audio cross-attention 추가.
    """
    def __init__(self, base_config, audio_config):
        super().__init__(base_config)
        
        self.audio_token_builder = AudioTokenBuilder(...)
        self.direction_encoder = DirectionEncoder(self.llm_hidden_dim)
        self.audio_grounding = AudioGroundingAdapter(self.vision_dim)
        
        # Freeze pretrained parts
        self._freeze_pretrained()
    
    def _freeze_pretrained(self):
        for p in self.vlm.parameters():
            p.requires_grad = False
        # Action expert는 stage에 따라 조절
    
    def forward(self, batch):
        # 1. Vision encoding (기존)
        vision_tokens = self.vision_encoder(batch['image'])
        
        # 2. Audio token building
        audio_events = batch['audio_events']  # SELD output
        audio_text_ids, dir_emb_positions, dir_embs = \
            self.audio_token_builder.build(audio_events)
        
        # 3. Language token embedding + direction embedding 주입
        lang_embeds = self.llm.embed(batch['instruction_ids'])
        audio_embeds = self.llm.embed(audio_text_ids)
        for pos, emb in zip(dir_emb_positions, dir_embs):
            audio_embeds[:, pos] = audio_embeds[:, pos] + emb
        
        # 4. Path C: vision-audio grounding
        vision_tokens = self.audio_grounding(vision_tokens, audio_embeds)
        
        # 5. Concat & decode (기존 SmolVLA logic)
        all_tokens = torch.cat([lang_embeds, audio_embeds, vision_tokens, state_token], dim=1)
        features = self.llm_decoder(all_tokens)
        
        # 6. Action expert
        action = self.action_expert(features)
        return action
```

### 6. Data Augmentation (`src/audio/augmentation.py`)

**학습 중 필수 augmentation** (robust policy를 위해):

```python
class SELDAugmentation:
    def __call__(self, events, p_each=0.3):
        # 1. Confidence perturbation: uncertainty reasoning 유도
        if random() < p_each:
            events = self.perturb_confidence(events, noise_std=0.2)
        
        # 2. Class confusion: visual evidence로 cross-check 능력 학습
        if random() < p_each * 0.5:  # 더 드물게
            events = self.swap_top_class(events)
        
        # 3. Direction noise: SELD 오차에 robust
        if random() < p_each:
            events = self.add_direction_noise(events, std_deg=5.0)
        
        # 4. Audio dropout: audio 없을 때도 동작
        if random() < 0.2:
            events = []  # 완전 제거
        
        # 5. Missing event: SELD가 일부 놓침 시뮬레이션
        if random() < p_each:
            events = self.drop_random_events(events, p=0.3)
        
        return events
```

**핵심**: 이 augmentation 없이는 LLM이 SELD 출력을 맹신하는 policy가 나옴. "SELD 불확실 → LLM common sense 활용" 능력을 명시적으로 학습시키는 열쇠.

---

## Training Pipeline

### Stage-by-Stage Training

#### Phase 0: Alignment Pretraining (선택, 권장)

```yaml
# configs/stage1_alignment.yaml
stage: alignment
trainable:
  - audio_token_builder (special token embeddings)
  - direction_encoder
  - audio_grounding (adapter only)
frozen:
  - vlm (all)
  - action_expert
objective: contrastive  # audio token ↔ related vision/language tokens
data: 100-200 episodes
steps: 3000-5000
batch_size: 32
lr: 1e-4
expected_time_rtx5090: 1-2 hours
```

**목적**: Audio module output이 LLM의 embedding space와 대략적으로 정렬되도록. 이 단계 없이 바로 Phase 1로 가면 초기 몇천 step 동안 action loss가 발산할 수 있음.

#### Phase 1: Main Training

```yaml
# configs/stage2_main.yaml
stage: main
trainable:
  - audio_token_builder
  - direction_encoder
  - audio_grounding
  - action_expert
frozen:
  - vlm (SigLIP + LLM)
objective: flow_matching  # SmolVLA 원 objective
data: 300-1500 episodes
steps: 30000-100000
batch_size: 32
optimizer:
  type: AdamW
  param_groups:
    - {params: audio_module, lr: 1e-4}      # 새 모듈: 빠르게
    - {params: action_expert, lr: 5e-5}     # 기존 모듈: 보통
  weight_decay: 1e-4
scheduler: cosine
warmup_steps: 500
expected_time_rtx5090: 10-30 hours
```

#### Phase 2: LLM LoRA (선택)

Spatial language reasoning이 약할 때만:

```yaml
# configs/stage3_lora.yaml
stage: lora
additional_trainable:
  - llm_lora (rank=16, upper 4-8 layers only)
frozen:
  - vision_encoder (SigLIP)
  - llm_lower_layers
optimizer:
  param_groups:
    - {params: audio_module, lr: 5e-5}
    - {params: action_expert, lr: 2e-5}
    - {params: llm_lora, lr: 1e-5}          # LoRA는 매우 conservative
steps: 20000-30000
batch_size: 16
expected_time_rtx5090: 15-20 hours
```

### Training Commands

```bash
# Phase 0: alignment
bash scripts/run_stage1.sh --config configs/stage1_alignment.yaml --output outputs/stage1

# Phase 1: main
bash scripts/run_stage2.sh \
  --config configs/stage2_main.yaml \
  --init_from outputs/stage1/best.ckpt \
  --output outputs/stage2

# Phase 2 (optional)
bash scripts/run_stage3.sh \
  --config configs/stage3_lora.yaml \
  --init_from outputs/stage2/best.ckpt \
  --output outputs/stage3
```

### Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| Batch size | 32 | RTX 5090 32GB에서 여유 있음 |
| Image resolution | 224 or 256 | SmolVLA 기본 (64 visual tokens) |
| Audio window | 2.5s | SELD sliding window |
| Audio hop | 0.25s | SELD prediction 주기 |
| Top-K events | 3 | uncertainty 전파 |
| Direction encoder dim | 128 internal | → LLM hidden dim으로 projection |
| Adapter layers (Path C) | 2 | 4 heads |
| Mixed precision | bf16 | Blackwell 안정적 |
| Gradient checkpointing | 선택 | 큰 batch 원할 때 |

---

## Evaluation

### Core Metrics

네 평가 매트릭스를 다층적으로:

```python
# src/eval/evaluator.py
class Evaluator:
    def evaluate_all(self, model, test_episodes):
        return {
            'overall_success_rate': ...,
            'clean_seld_success': self.eval_with_clean_seld(...),
            'noisy_seld_success': self.eval_with_noisy_seld(...),
            'spatial_reasoning': self.eval_spatial_instructions(...),
            'audio_ambiguous_vision': self.eval_audio_only_disambiguation(...),
            'common_sense_recovery': self.eval_seld_failure_recovery(...),
        }
```

### Evaluation Conditions

1. **Clean SELD**: Ground truth 수준 SELD → baseline 성공률
2. **Noisy SELD**: Confidence 강제 저하, class 10–20% 의도적 오분류 → common sense 활용 측정
3. **Spatial reasoning**: "오른쪽의 물체", "사이렌 반대편" 같은 relational instruction
4. **Audio-only disambiguation**: 시각적으로 동일한 물체 중 audio로만 구별
5. **No audio fallback**: Audio 없을 때 성능 유지 (regression 없어야 함)

### Ablation Studies

필수 ablation:
- **No audio**: baseline SmolVLA
- **Path A only**: language stream 주입만
- **Path C only**: vision cross-attention만
- **Path A + C**: full method
- **Top-1 vs Top-K**: uncertainty 전파 효과
- **Text-only audio** (베이스라인): 순수 텍스트 인젝션 ("siren at 30 degrees")

### Probing (LLM Knowledge 활용 확인)

```python
# src/eval/probing.py
def probe_common_sense_usage(model, scene, audio_event):
    """
    LLM의 pretrained knowledge가 실제로 쓰이는지 검증.
    
    예: audio=siren, low conf. Scene에 alarm clock + book.
    - 올바른 policy: alarm clock 선택 (common sense)
    - LLM knowledge 미활용 policy: random 또는 bias
    """
    ...
```

---

## RTX 5090 Resource Planning

- **VRAM**: 32GB
  - SmolVLA 450M + optimizer state (bf16): ~10GB
  - Audio module: +1–2GB
  - Batch 32 activations: ~8GB
  - **여유**: batch 64까지 가능
- **Training time estimates**:
  - Phase 0: 1–2h
  - Phase 1 (30k steps): 10–15h
  - Phase 1 (100k steps): 1.5–2일
  - Phase 2 (30k steps): 15–20h
- **Tips**:
  - `bf16` mixed precision (Blackwell 최적화)
  - Flash Attention 2 활성화
  - `num_workers=8-16`, SSD 권장
  - SELD 사전 계산으로 data loading bottleneck 제거

---

## Failure Modes & Mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Catastrophic forgetting | 기존 task 성능 저하 | Audio-free episodes 20–30% 포함 |
| Audio dominance | Vision 무시, audio만 참조 | Audio dropout 20–30% |
| Special token 학습 부족 | `[AUDIO]` 토큰 학습 더딤 | 유사 vocab embedding으로 초기화 |
| Direction embedding explosion | 초기 학습 불안정 | Small-scale init (std=0.01) |
| SELD 맹신 | Noisy SELD에서 성능 급락 | Confidence/class augmentation 적극 사용 |
| Phase 1 초기 발산 | Action loss 튐 | Phase 0 alignment 먼저 수행 |

---

## Milestones & Timeline

| Week | Milestone | Deliverable |
|---|---|---|
| 1–2 | Baseline 재현 | SmolVLA vanilla fine-tune 성공 |
| 3–4 | SELD 세팅 + 소량 데이터 수집 | SELD 추론 파이프라인 동작 |
| 5–6 | Audio module 구현 + Phase 0 | Alignment 학습 완료 |
| 7–8 | Main dataset 수집 완료 | 500+ 에피소드 |
| 9–10 | Phase 1 학습 + 1차 eval | 결과 분석 |
| 11 | Ablation studies | Full comparison table |
| 12 | Failure analysis + 최종 실험 | 논문/보고서 초안 |

---

## Reference Configurations

### Minimal Dev Config (빠른 iteration용)

```yaml
data:
  episodes: 50
  audio_augmentation: True
training:
  steps: 5000
  batch_size: 16
  stage: stage2_only
  skip_alignment: True
eval:
  interval: 500
```

### Production Config (본 실험용)

```yaml
data:
  episodes: 800
  audio_augmentation: True
  augmentation_prob: 0.3
training:
  stages: [alignment, main]
  alignment_steps: 5000
  main_steps: 50000
  batch_size: 32
eval:
  interval: 2000
  metrics: [overall, clean_seld, noisy_seld, spatial, ablation]
```

---

## Open Questions & Future Work

1. **SELD 모델 선택**: SELDnet vs EINv2 vs CST-Former — 정확도/latency trade-off
2. **Reasoning trace annotation**: 10–20% 에피소드에 CoT 추가 시 개선폭
3. **Camera FoV 밖 audio**: Path A만으로 충분한지, Path C 확장 필요한지
4. **더 큰 VLA 확장**: π0, OpenVLA로 포팅 시 어떤 수정 필요한지
5. **Online adaptation**: 배포 후 SELD 성능 drift 대응

---

## Key Design Principles (요약)

1. **Hybrid injection**: Path A (LLM reasoning) + Path C (spatial grounding) 병행
2. **Preserve pretrained weights**: Zero-init gated adapter, small-scale embedding init
3. **Leverage LLM vocab**: Class명은 텍스트 토큰으로 (common sense 활성화)
4. **Continuous direction embedding**: 각도는 sin/cos + MLP (precision)
5. **Uncertainty as first-class citizen**: Top-K + confidence + augmentation
6. **Staged training**: Alignment → main → (optional) LoRA
7. **Separated learning rates**: 새 모듈 > 기존 모듈 > LoRA
8. **Graceful degradation**: Audio dropout으로 audio 없이도 동작하는 policy
