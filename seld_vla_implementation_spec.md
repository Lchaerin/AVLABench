# SELD→VLA Audio Injection: Implementation Spec

구현 대상: SELD(sound event localization & detection) 출력을 VLA(π0 우선, SmolVLA 대체 경로)에 주입해
오디오 조건 조작 태스크(예: "소리 나는 라디오를 꺼라")를 수행하는 전체 파이프라인.
이 문서 하나로 구현 가능하도록 작성됨. 모호한 부분은 `config.yaml`의 기본값을 따른다.

---

## 0. 시스템 개요

```
mic array ──► SELD(multi-ACCDOA) ──► [M1 집계+트래킹] ──► AudioEvent[]
                                                            │
   fixed 3rd-person cam (mic와 동일 pose) ──intrinsics──► [M2 az/el→(u,v) 투영, FOV 밖 drop]
                                                            │
                                          [M3 직렬화] <audio> radio <ASLOT> ; ... </audio>
                                                            │
  images ─► ViT ─┐                                          ▼
                 ├─► LLM input_embeds  ◄── [M4 SlotEncoder가 <ASLOT> 위치 임베딩 치환]
  instruction ───┘                                          │
                                              [M5 VLA: π0 / SmolVLA(+LoRA)]
                                                            │
                                              flow-matching action expert ─► actions
```

- 핵심 설계: 클래스=텍스트 토큰(LLM 사전지식 활용), 위치·음량=이미지 좌표 기반 연속 슬롯 임베딩.
- DoA는 (az, el)을 **고정 3인칭 카메라의 이미지 좌표 (u,v)로 투영**해 전달한다(카메라·마이크 동일 pose).
  화면 밖 이벤트는 기본 drop.
- 침묵도 항상 표현: `<audio> silence </audio>`. 센서 부재는 `<audio> unavailable </audio>`로 구분.

## 1. Repo 구조

```
seld_vla/
  config.yaml
  audio/
    seld_interface.py   # M1: SELD 출력 파싱, 집계, 트래킹, 히스테리시스
    projection.py       # M2: az/el → (u,v)
    events.py           # AudioEvent dataclass, 정렬/절삭
  serialization.py      # M3: 프롬프트 조립
  model/
    slot_encoder.py     # M4: Fourier+MLP+LN+gate
    injection.py        # M5: <ASLOT> 임베딩 치환, π0/SmolVLA 어댑터
  data/
    synthesis.py        # M6: 시뮬 기하학→라벨, 공간오디오 렌더링(pyroomacoustics)
    dataset.py          # LeRobot 포맷 확장 로더
  training/
    stage_a.py          # M7a: grounding 사전학습
    stage_b.py          # M7b: 정책 파인튜닝(커리큘럼, dropout, 믹스)
  inference/
    loop.py             # M8: 청크 단위 추론 루프
  eval/
    protocol.py         # M9: 티어별 평가, 지표
  tests/                # 각 마일스톤 단위테스트
```

## 2. config.yaml (기본값 = 이 스펙의 정답)

```yaml
audio:
  seld_hop_s: 0.1            # SELD 프레임 hop
  agg_window_s: 1.0          # 집계 윈도우(직전 N초)
  conf_on: 0.5               # 트랙 등장 임계
  conf_off: 0.3              # 트랙 소멸 임계 (히스테리시스)
  min_frames: 3              # 등장 확정에 필요한 최소 활성 프레임
  match_deg: 20.0            # 프레임 간 동일 트랙 매칭 각거리(°)
  k_max: 4                   # 직렬화 최대 이벤트 수
  energy_db_range: [30, 90]  # dB SPL → [0,1] 정규화 구간
  offscreen_policy: drop     # drop | mark
  seld_convention: dcase     # dcase | custom (아래 §4 참고)
camera:
  name: front_fixed          # 마이크와 동일 pose인 3인칭 고정 카메라
  intrinsics: {fx: 0.0, fy: 0.0, cx: 0.0, cy: 0.0}  # 캘리브레이션 값 주입
  width: 0
  height: 0
  R_mic2cam: identity        # 동일 pose 가정; 미세 오프셋 시 3x3 회전 기입
slot:
  fourier_L: 6               # (u,v) 각각에 L밴드
  hidden: 256
  gate_init: 0.1
serialization:
  slot_token: "<ASLOT>"
  block_open: "<audio>"      # 특수토큰 아님, 일반 텍스트로 토크나이즈
  block_close: "</audio>"
  sep: " ; "
  silence_text: "silence"
  unavailable_text: "unavailable"
train:
  beta_lang: 1.0
  grounding_batch_ratio: 0.25   # Stage B에서 grounding 배치 비율 (action:grounding = 3:1)
  modality_dropout_p: 0.1
  jitter:                       # 노이즈 커리큘럼 (step 비율 구간)
    phase1: {until: 0.3, sigma_az_deg: 0.0, sigma_el_deg: 0.0}
    phase2: {until: 0.7, sigma_az_deg: 15.0, sigma_el_deg: 8.0}   # 0→값 선형 램프
    phase3: {until: 1.0, source: seld}  # 실제 SELD 출력; 없으면 phase2 유지
  lora: {r: 32, alpha: 32, targets: [q_proj, k_proj, v_proj, o_proj]}
model:
  backbone: pi0              # pi0 | smolvla
```

## 3. M1 — SELD 인터페이스·집계·트래킹 (`audio/seld_interface.py`)

입력(프레임 단위, SELD 모델이 제공):
```python
SELDFrame = {t: float, detections: [{class_id:int, az_deg:float, el_deg:float,
                                     activity:float, energy_db:float|None}]}
```
- multi-ACCDOA면 activity = DoA 벡터 노름. energy_db 부재 시 어레이 채널 RMS→dB로 대체(캘리브레이션 상수 1회 측정), 그것도 없으면 `energy = activity`로 프록시(config 주석 명시).

처리:
1. **트래킹**: 직전 윈도우 트랙과 (class 동일 ∧ 각거리 < `match_deg`)이면 동일 트랙. 각거리 = arccos(d1·d2), d는 §4의 방향벡터.
2. **히스테리시스**: 비활성→활성 전이는 `activity ≥ conf_on`이 `min_frames` 이상 지속될 때. 활성→비활성은 `activity < conf_off`가 `min_frames` 지속.
3. **집계**(활성 트랙별, 윈도우 내): az/el = 원형 평균(circular mean; sin/cos 평균 후 atan2), energy = 평균, confidence = 평균 activity를 [0,1] 클립.

출력:
```python
@dataclass
class AudioEvent:
    class_name: str      # class_id → 텍스트 매핑 테이블 (예: "radio","alarm","phone","speech",...)
    az_deg: float; el_deg: float
    energy: float        # [0,1], clip((db-30)/60, 0, 1)
    confidence: float    # [0,1]
    uv: tuple|None       # M2가 채움; None=화면 밖
```

## 4. M2 — DoA→이미지 좌표 투영 (`audio/projection.py`)

**좌표 규약(중요, 단위테스트 필수):**
- `seld_convention: dcase` = DCASE 표준: x전방, y좌측, z상방. az ∈ [-180,180), **좌측(반시계) 양수**. el ∈ [-90,90], 상방 양수.
- 카메라 = OpenCV: X우측, Y하방, Z전방.

```python
def doa_to_uv(az_deg, el_deg, K, wh, R_mic2cam=None):
    az, el = radians(az_deg), radians(el_deg)
    # DCASE frame 방향벡터
    d_f = np.array([cos(el)*cos(az),   # x fwd
                    cos(el)*sin(az),   # y left
                    sin(el)])          # z up
    # DCASE → OpenCV: X=-y_f(right), Y=-z_f(down), Z=x_f(fwd)
    d_c = np.array([-d_f[1], -d_f[2], d_f[0]])
    if R_mic2cam is not None: d_c = R_mic2cam @ d_c   # 미세 pose 오프셋 보정(회전만; far-field라 t 무시)
    if d_c[2] <= 1e-6: return None                    # 카메라 후방
    u = K.fx * d_c[0]/d_c[2] + K.cx
    v = K.fy * d_c[1]/d_c[2] + K.cy
    W, H = wh
    if not (0 <= u < W and 0 <= v < H): return None   # FOV 밖
    return (u / W, v / H)                             # 정규화 [0,1)
```
- 이미지에 렌즈 왜곡이 있으면 **학습·추론 이미지 자체를 undistort**하고 위 핀홀만 사용(왜곡모델 이중적용 금지).
- `uv is None`인 이벤트: `offscreen_policy=drop`이면 제외(기본). `mark`면 슬롯 없이 `"radio (offscreen)"` 텍스트만.
- **단위테스트**: az=0,el=0 → (cx/W, cy/H). az=+30°(좌) → u < cx/W (좌측). el=+20°(상) → v < cy/H. 후방/화면밖 → None.

## 5. M3 — 직렬화 (`serialization.py`)

프롬프트 조립(π0/PaliGemma 프리픽스 텍스트 부분):
```
{audio_block} {language_instruction}
```
audio_block 규칙:
- 이벤트 있음: `<audio> radio <ASLOT> ; radio <ASLOT> ; alarm <ASLOT> </audio>`
  - **energy 내림차순 정렬**, `k_max`개 초과분 절삭.
- 이벤트 없음(정상 관측): `<audio> silence </audio>`  ← 슬롯 없음
- 오디오 센서 부재/드롭아웃: `<audio> unavailable </audio>`
- 블록은 **항상 존재**한다(구조 일관성). `silence`/`unavailable`은 새 특수토큰이 아니라 일반 어휘 단어.
- `<ASLOT>`만 토크나이저에 신규 추가(1개). `tokenizer.add_tokens(["<ASLOT>"]); model.resize_token_embeddings(...)`.
  - LM loss 계산 시 `<ASLOT>` 위치 label = -100 (절대 타깃으로 학습 금지).

## 6. M4 — SlotEncoder (`model/slot_encoder.py`)

```python
def fourier(p, L):  # p∈[0,1] 스칼라
    return concat([f(2**k * pi * p) for k in range(L) for f in (sin, cos)])   # 2L dim

class SlotEncoder(nn.Module):
    # in_dim = 2*2L (u,v fourier) + 2 (u,v raw) + 1 (energy) + 1 (confidence) = 4L+4  (L=6 → 28)
    def __init__(self, d_model, L=6, hidden=256, gate_init=0.1):
        self.mlp = MLP(4*L+4, hidden, d_model, act=GELU, layers=2)
        self.ln  = LayerNorm(d_model)
        self.a   = nn.Parameter(torch.tensor(atanh(gate_init)))   # gate
    def forward(self, u, v, energy, conf):
        x = concat([fourier(u,L), fourier(v,L), [u, v, energy, conf]])
        return tanh(self.a) * self.ln(self.mlp(x))                # (d_model,)
```
- `d_model = llm.config.hidden_size` (π0/Gemma-2B: 2048, SmolVLA/SmolVLM2: 모델에서 읽기).
- **gate 근거**: 초기 미정렬 임베딩이 사전학습 attention을 교란하지 않도록 작게 시작(Flamingo tanh-gating 원리). gate/LN 제거는 ablation 항목.

## 7. M5 — 임베딩 주입 (`model/injection.py`)

```python
ids = tokenizer(prompt_with_slots)                    # <ASLOT> 포함
emb = llm.embed_tokens(ids)                           # (T, d)
for pos, ev in zip(slot_positions(ids), events):      # 좌→우 = energy 내림차순, 개수 일치 assert
    emb[pos] = slot_encoder(ev.uv[0], ev.uv[1], ev.energy, ev.confidence)
# 이후 emb를 통상 경로(이미지 토큰과 concat 등)로 전달
```
백본별 경로:
- **π0 (기본)**: PaliGemma 프리픽스(이미지+텍스트)에 위 방식 적용. 파인튜닝은 π0 표준 레시피(전체 or LoRA) — VLM이 학습되므로 연속 슬롯과 호환.
- **SmolVLA (대체)**: 기본 레시피는 VLM 완전 동결 → **동결 VLM에 연속 슬롯 주입 금지**(무시/불안정 위험). 반드시 (a) VLM에 LoRA(`train.lora`) 적용 후 슬롯 주입, 또는 (b) 슬롯 없이 텍스트 직렬화(§11 B2)로 대체. config `model.backbone`으로 분기.
- 오디오 블록 토큰 수 ≈ 이벤트당 3~4 + 델리미터 → 최대 ~20토큰. 시퀀스 예산 문제 없음.

## 8. M6 — 데이터 합성·스키마 (`data/`)

**라벨 자동화(시뮬)**: 시뮬레이터에서 음원 부착 물체의 3D 위치 + 카메라 pose → 정답 az/el(oracle) 및 GT bbox 자동 산출. 공간오디오 파형이 필요한 경우(실 SELD 학습/평가용) pyroomacoustics로 어레이 IR 컨볼루션 렌더링; **정책 학습만이면 파형 없이 oracle 라벨로 충분**(SELD는 추론 시에만 개입).

**데이터셋 스키마(LeRobot 포맷 확장)**: 프레임별 필드 추가
```python
frame["audio_events"] = [{class_name, az_deg, el_deg, energy, confidence}]  # 빈 리스트 = silence
meta["camera_intrinsics"], meta["R_mic2cam"], meta["seld_convention"]       # 투영 재현용 원시값 저장
```
- (u,v)는 저장하지 않고 로더에서 M2로 매번 계산(카메라 재캘리브레이션 대응).

**에피소드 믹스(Stage B)**:
| 유형 | 비율 | 목적 |
|---|---|---|
| 오디오 필수(Tier1–4) | 45% | 주 능력 |
| 방해음(소리 존재하나 태스크 무관: "알람 울리는 중 컵 집기") | 25% | 소리 추종 지름길 차단 |
| 침묵/오디오 무관 조작 | 30% | silence 표현 학습, 기본 능력 유지 |
- 위에 더해 학습 시 확률 `modality_dropout_p=0.1`로 블록을 `unavailable`로 치환.

## 9. M7 — 학습

**Stage A — Grounding 사전학습 (액션 없음, 시뮬 합성으로 대량 생산)**
입력 = 이미지 + audio_block(슬롯 포함), 타깃 = 텍스트. 태스크 4종 균등 믹스:
1. `"Which object is making sound?"` → `"the radio on the left"` (자유 텍스트)
2. `"detect the sounding object"` → `<locYYYY><locXXXX><locYYYY><locXXXX> radio`
   (PaliGemma 네이티브 detect 포맷: y_min,x_min,y_max,x_max, 0–1023 정규화, GT bbox 사용)
   → **이 태스크가 슬롯의 (u,v)↔시각 공간 대응을 직접 지도함. Tier 2의 핵심 선행 능력.**
3. `"Is anything making sound?"` → `"no"` (silence 샘플)
4. `"Which sound is louder?"` → `"the alarm"` (2+ 이벤트 샘플)
- 학습 대상: SlotEncoder + `<ASLOT>` 임베딩 + LoRA(또는 π0 전체). loss = 표준 LM CE.
- SmolVLA 경로: SmolVLM2에는 loc 토큰이 없음 → 태스크 2를 텍스트 좌표(`"at (0.32, 0.61)"`)로 대체.

**Stage B — 정책 파인튜닝**
- 배치 믹스: action 배치 : grounding 배치 = 3 : 1 (`grounding_batch_ratio=0.25`).
  action 배치 loss = flow-matching, grounding 배치 loss = LM CE. (한 배치에 이중 loss 아님 — 메모리·구현 단순화. FuSe 교훈: 언어 접지 손실이 없으면 새 모달리티 무시.)
- **노이즈 커리큘럼**(`train.jitter`): oracle → az/el에 가우시안 지터(σ 선형 램프, **투영 전 각도 공간에서** 적용) → 가능하면 실제 SELD 출력. energy에도 σ=0.05 지터.
- 옵티마이저/스케줄은 각 백본 공식 파인튜닝 기본값 사용. SlotEncoder lr은 백본 lr×10.

## 10. M8 — 추론 루프 (`inference/loop.py`)

```python
tracker = SELDTracker(cfg)
while not done:
    events = tracker.update(seld_stream.frames_since_last_call())  # M1 (히스테리시스 상태 유지)
    events = [project(e) for e in events if project(e)]            # M2 (drop policy)
    prompt = serialize(events, instruction)                        # M3 — 매 청크마다 갱신
    actions = vla.infer(images, prompt, state)                     # 프리픽스 재계산 (1–2Hz면 비용 무시 가능)
    execute(actions)   # action chunk (~1s)
```
- 소리 멈춤 → 다음 청크에서 `silence` 블록 → 정책이 완료를 인지(끄기 태스크의 종료 신호).
- 지속음 가정. 청크 길이보다 짧은 transient는 놓칠 수 있음(알려진 한계, 태스크 설계에서 지속음 사용).

## 11. M9 — 평가 프로토콜 (`eval/protocol.py`)

**태스크 티어**(각 티어 1–2개 태스크):
- T1 분류만: "삐 소리 나는 가전 끄기" (단일 음원)
- T2 DoA 필수(주력): "소리 나는 라디오 끄기" — **시각적으로 동일한 라디오 2대**, DoA로만 구분 가능
- T3 다중음원+방해음: "왼쪽에서 음악 나오는 라디오 끄기" (라디오·알람 동시)
- T4 조합: "더 시끄러운 라디오 끄기" / "소리 안 나는 라디오 켜기"(negation)

**규모·지표**:
| | 시뮬 | 실기 |
|---|---|---|
| ablation | 50 rollouts/task | — |
| 최종 | 500 rollouts/task × 3 seeds | 20–25 trials/task |
| 지표 | 성공률 | 성공률 + 부분점수(대상선택 0.5 + 조작 0.5) |
- 모든 비교는 동일 초기 배치 A/B. **oracle DoA vs SELD DoA 이중 평가**로 오차 원인 분리.
- 추가 지표: disambiguation accuracy(T2), SELD 각오차 vs 성공률 상관.

**Baselines / Ablations**:
- B0 vision-only(오디오 블록 unavailable 고정)
- B1 loc-token 점 직렬화(π0만): `radio at <locVVVV><locUUUU>` (y,x 순; 슬롯 없음)
- B2 평문 직렬화: `radio at azimuth 40, elevation 10, loud` (15° bin; SmolVLA 동결 경로 겸용)
- B3 제안(슬롯) — vs B1/B2가 논문의 핵심 비교
- A1 Fourier 제거(u,v raw만) / A2 gate·LN 제거 / A3 grounding loss 제거 / A4 방해음 믹스 제거 / A5 silence 블록 제거(이벤트 없을 때 블록 생략)

## 12. 구현 순서 & 완료 기준

1. M2 투영 + 단위테스트(§4 케이스) → 2. M1 트래커(합성 프레임 시퀀스로 히스테리시스 테스트) →
3. M3+M5 직렬화·주입(스냅샷 테스트: 토큰열/슬롯 위치 assert) → 4. M4 SlotEncoder(shape·gate 초기값 테스트) →
5. M6 합성 파이프라인(렌더링된 시뮬 프레임에 (u,v) 점 오버레이 시각 검증 스크립트 필수) →
6. Stage A 학습 → 검증: held-out에서 태스크2 bbox IoU@0.5 ≥ 0.7 달성 후 진행 →
7. Stage B → 8. M9 평가.

## 13. 함정 체크리스트

- [ ] `<ASLOT>`이 LM label에 노출되지 않는가(-100 마스킹)
- [ ] 좌표 규약: SELD의 az 부호(DCASE=좌측 양수)와 §4 변환 일치 확인 — **가장 흔한 버그**
- [ ] 이미지 undistort와 핀홀 투영의 일관성(왜곡 이중적용 금지)
- [ ] 슬롯 개수 == `<ASLOT>` 개수 assert (직렬화·주입 사이 불일치 시 즉시 실패)
- [ ] 침묵 비율이 데이터에서 20–40% 유지(과다 시 오디오 무시 편향)
- [ ] SmolVLA에서 VLM 동결 상태로 슬롯 주입하지 않았는가
- [ ] 지터는 투영 **전** 각도 공간에서 적용했는가
- [ ] 히스테리시스 상태가 에피소드 시작 시 리셋되는가
- [ ] 평가에서 두 라디오가 시각적으로 완전 동일한가(LED 등 시각 단서 제거)
