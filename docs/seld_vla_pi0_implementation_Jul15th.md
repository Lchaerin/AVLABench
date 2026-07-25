# SELD-VLA (pi0) 구현 문서 — SlotEncoder 기반 오디오 조건부 정책

> 대상: `seld_vla_implementation_spec.md`의 오디오→VLA 주입 파이프라인을 **pi0/openpi 백본 + 오라클**로
> 구현한 결과. 실제 SELD 모델은 아직 개발 중이라, 현재는 시뮬레이터 GT를 쓰는 **oracle**로 학습·평가한다.
> 태스크는 라디오 3종(`select_radio` / `select_radio_two` / `select_radio_silent`).

---

## 0. 모델 구조 개요

핵심 아이디어(스펙과 동일): **"소리의 정체(class)는 언어 토큰으로, 소리의 위치·크기는 연속 임베딩으로"** VLA에 준다.
- **class** → 실제 텍스트 단어로 프롬프트에 넣어 LLM의 사전지식을 활용한다.
- **위치(DoA)** → 고정 3인칭 카메라 이미지 좌표 `(u, v)`로 **투영**해 시각적으로 접지(grounding)한다.
- **크기(energy/loudness)** → `[0,1]` 스칼라로 넘기고, 동시에 `loud/moderate/quiet` 단어로도 표현한다.

```
                                   ┌─────────────────────────────────────────────┐
 오라클(시뮬 GT)                    │  프롬프트(텍스트)                              │
   radio 위치/클래스/에너지 ──┬──►  │  "<audio> radio loud ; alarm quiet </audio>   │
                              │     │   primitive: Press the button ... sound."     │
                              │     └──────────────────┬──────────────────────────┘
                              │                        │ tokenize
                              │                        ▼
   az/el ──[M2 투영]──► (u,v) │            PaliGemma 임베딩 (image + language)
                              │                        │
                              └──► audio_slots[K,6]     │  ← 언어 블록 뒤에 append
                                  =(u,v,energy,conf,    │
                                    class_id,present)   ▼
                                        [M4 SlotEncoder] ─► 슬롯당 1 토큰
                                        class_emb(cid)·√W + SlotEncoder(u,v,energy,conf)
                                                          │
                                                          ▼
                                          pi0 (PaliGemma + action expert)
                                                          │
                                              flow-matching ─► action chunk
```

- 백본: **pi0** (openpi PyTorch 경로). PaliGemma(비전 SigLIP + Gemma-2B LLM) + flow-matching action expert.
- 파인튜닝: **LoRA**(기본). PaliGemma base는 동결, LLM attention/MLP에 LoRA 어댑터, action expert와 오디오
  모듈(SlotEncoder·class embedding)은 전체 학습. → 스펙 §7의 "pi0는 VLM이 학습되므로 연속 슬롯 주입과
  호환" 조건을 LoRA가 충족한다(완전 동결 VLM에 연속 슬롯을 주입하면 무시/불안정 위험이 있음).

---

## 1. 데이터 흐름 (end-to-end)

```
trajectory_generation.py         convert_hdf5_to_lerobot.py       openpi 데이터 파이프라인
 (conda, VLABench sim)             (openpi venv, lerobot 0.1.0)     (학습/추론 공용 transform)
──────────────────────────       ──────────────────────────       ──────────────────────────
 oracle GT 추출:                  observation.audio.*:              InjectAudioBlockText:
   az/el (SLED 부호)                azimuth_deg, elevation_deg,       class+loudness → 프롬프트
   energy (거리 기반)               confidence(=1.0), class_id,      BuildAudioSlotsUV:
   cam_fovy                         energy, uv[K,2]                    audio_slots[K,6] 패킹
 → meta_info/oracle_audio        (uv는 az/el을 M2로 투영)          → PI0Pytorch._embed_audio_slots
```

한 프레임이 VLA에 도달하기까지의 필드 변환:

| 단계 | 산출물 | 위치 |
|---|---|---|
| 오라클 GT | `az_deg, el_deg, energy, distance, cam_fovy_deg` | `src/audio/oracle_sled.py` |
| 변환(저장) | `observation.audio.{azimuth_deg, elevation_deg, confidence, class_id, energy, uv}` | `src/data/convert_hdf5_to_lerobot.py` |
| 텍스트 주입 | `"<audio> ... </audio> {instruction}"` | `openpi .../vlabench_policy.py: InjectAudioBlockText` |
| 슬롯 패킹 | `audio_slots [K,6]` | `vlabench_policy.py: BuildAudioSlotsUV` |
| 모델 주입 | 언어 블록 뒤 append 토큰 | `openpi .../pi0_pytorch.py: _embed_audio_slots` |

---

## 2. 좌표 규약 (스펙 §13의 "가장 흔한 버그")

두 azimuth 부호 규약이 공존한다.

| 규약 | 정의 | 사용처 |
|---|---|---|
| **HRTF** | 0=정면, +90=**좌**, −90=우 | `binaural_engine.compute_listener_relative_direction` |
| **SLED** | `az_sled = -az_hrtf` → **우측이 +** | 데이터셋에 저장되는 값(`observation.audio.azimuth_deg`) |
| **DCASE** | 좌측이 + (스펙 §4 `doa_to_uv`의 기준) | 스펙 원문 수식 |

데이터셋은 **SLED(우측 +)**로 저장하므로, 투영할 때 반드시 `convention="sled"`를 넘겨 내부에서
`az → -az`로 바꿔 DCASE 수식을 재사용한다. 이 부호 하나가 틀리면 좌우가 뒤집혀 학습이 조용히 망가진다.

> 단위테스트(`tests/test_projection.py`)로 못박음: `sled az=+15° → u>0.5(우)`, `dcase az=+15° → u<0.5(좌)`,
> 두 값이 정확히 대칭(`sled(+15)==dcase(-15)`), 후방/화면밖 → `None`.

---

## 3. 파트별 상세 & 구현 이유

### 3.1 M2 — DoA→이미지 좌표 투영 `src/audio/projection.py`

- `CameraIntrinsics.from_fovy(fovy, W, H)`: MuJoCo 카메라의 **수직 FOV(fovy)**로 핀홀 intrinsics 생성.
  정사각 픽셀 가정(`fx=fy=(H/2)/tan(fovy/2)`), 주점은 이미지 중심. 렌더가 정사각(224×224)이라 정규화
  `(u,v)`는 해상도 불변.
- `doa_to_uv(az, el, K, wh, R_mic2cam=None, convention)`: DCASE 방향벡터 → OpenCV 카메라 좌표
  `(right, down, fwd)` → 핀홀 투영 → 정규화 `[0,1)`. 후방(`z≤0`)·FOV 밖은 `None`(스펙의 `offscreen_policy=drop`).
- **왜 (u,v)인가**: 각도(az/el)를 그대로 주는 것보다, "이미지의 어디"로 접지하면 비전 토큰과 직접 대응되어
  멀티소스 구분(스펙 T2: 시각적으로 동일한 라디오 2대) 학습이 쉬워진다.

### 3.2 M4 — SlotEncoder `src/audio/slot_encoder.py`

- 입력: `Fourier(u,L) ⊕ Fourier(v,L) ⊕ [u, v, energy, conf]` (in_dim = 4L+4, L=6 → 28).
- 구조: `MLP → LayerNorm → tanh-gate`. 게이트는 `a=atanh(0.1)`로 초기화해 **초기 출력이 작게**(≈0.1) 시작.
- **왜 gate/Fourier인가**: (1) Fourier는 저차원 좌표를 고주파까지 표현해 위치 민감도↑(NeRF/positional
  encoding과 동일 원리). (2) tanh-gate(Flamingo 방식)는 학습 초기 미정렬 임베딩이 사전학습된 attention을
  교란하지 않게 하고, 게이트가 학습되며 서서히 개입한다. `gate=0`이면 슬롯 임베딩이 정확히 0(테스트로 검증).

### 3.3 M3 — 오디오 블록 직렬화 `src/audio/slot_serialization.py`

- `build_prompt(events, instruction) → "{audio_block} {instruction}"`.
- 블록 규칙: 이벤트 있음 `"<audio> radio loud ; alarm quiet </audio>"`, 없음 `"<audio> silence </audio>"`,
  센서 부재 `"<audio> unavailable </audio>"`. **블록은 항상 존재**(구조 일관성).
- energy 내림차순 정렬 + `k_max` 절삭, 화면밖(`uv=None`) drop, `loudness_word`(≥0.66 loud / ≥0.33 moderate / else quiet).
- `include_slot_token` 옵션: 스펙 원안의 **inline `<ASLOT>`** 마커를 넣을지 여부. **pi0 경로는 False**(3.6 참고).
- **왜 class를 텍스트로 두나**: LLM은 "radio", "dog", "alarm" 같은 단어의 의미를 이미 안다. 이를 그대로
  쓰면 instruction의 클래스어("...playing the dog sound")와 오디오 블록의 클래스어가 같은 임베딩 공간에서
  매칭되어, 멀티소스에서 "어떤 소리의 라디오인가"를 언어로 접지할 수 있다.

### 3.4 오라클 energy/loudness `src/audio/oracle_sled.py`

- `energy_from_distance(dist, level_db)`: 역거리 SPL 법칙(`75dB @1m`, `-20log10(d)`)을 스펙의
  `[30,90]dB → [0,1]`로 정규화. SLED가 새로 넘겨주는 loudness를 오라클로 모사.
- `extract_episode_gt`가 소스별 `energy`와 `cam_fovy_deg`를 기록(후자는 (u,v) 투영용 intrinsics 재구성에 필요).
- `build_oracle_topk(_batched)`는 `energy` 배열을 출력(σ=0.05 지터 옵션, 스펙 §9).
- **왜 거리 기반인가**: 시뮬엔 실제 dB가 없다. 거리 감쇠는 물리적으로 타당하고, 무엇보다 **활성(소리남,
  energy>0)과 침묵(energy=0)을 분리**해 `silent` 태스크의 신호가 된다.

### 3.5 데이터셋 스키마 `src/data/convert_hdf5_to_lerobot.py`

- 신규 컬럼: `observation.audio.energy [K]`, `observation.audio.uv [K,2]`.
- 오라클 분기에서 **저장 시점에** 각 슬롯의 clean GT az/el을 `(u,v)`로 투영(화면밖 → `(-1,-1)` sentinel).
  intrinsics는 `oracle_mode.json`에 함께 저장(재현성).
- `confidence`는 `1.0`으로 저장("GT는 확실"), energy는 GT clean 값.
- **왜 저장 시점 투영인가**: 스펙은 "로더에서 매번 투영"을 권하지만, 오라클은 카메라·소스가 에피소드 내
  정적이고 openpi 데이터 transform(numpy, 프레임 단위)이 meta에 접근하기 번거롭다. 저장 시 한 번 투영하면
  torch 모델은 단순해지고 재현성도 유지된다(intrinsics를 meta에 남겨 재캘리브레이션 대응 가능).

### 3.6 pi0 주입 `third_party/openpi` (submodule, branch `pi05`)

새 오디오 모드 **`audio_mode="slots_uv"`** + `Pi0Config.audio_slot_uv=True`.

- **`pi0_pytorch.py`**: `SlotEncoder`를 vendored(서브모듈이 repo `src/`에 학습 시점 import 의존하지 않도록
  복제; canonical은 `src/audio/slot_encoder.py`, 동기화 필요). `_embed_audio_slots`가 슬롯 마지막 차원으로
  분기: `[K,6]`이면 `class_embedding(cid)·√W + SlotEncoder(u,v,energy,conf)`, 아니면 기존 az/el DirectionEncoder.
- **`vlabench_policy.py`**: `InjectAudioBlockText`(class+loudness→프롬프트, 컬럼 pop 안 함) +
  `BuildAudioSlotsUV`(`[K,6]=(u,v,energy,conf,class_id,present)` 패킹). `present`는 class≥0 ∧ 화면안일 때 1.
- **`training/config.py`**: slots_uv가 energy/uv 컬럼을 surface + 두 transform 삽입. TrainConfig 2개
  (`pi0_ft_vlabench_seld_uv_three_radio[_lora]`).

#### 설계상의 절충: inline `<ASLOT>` 대신 **append 토큰**

스펙 원안(M5)은 프롬프트에 `<ASLOT>` 토큰을 넣고 그 위치 임베딩을 SlotEncoder 출력으로 **치환**한다.
그러나 openpi의 PaliGemma 토크나이저는 동결·고정이라 새 vocab 추가는 임베딩 resize 수술이 필요하고
이 세션에서 end-to-end 검증이 어렵다. 그래서 pi0에서는:

- **표현은 스펙 그대로**: `SlotEncoder(u,v,energy,conf)` + class=텍스트.
- **주입 방식만 변경**: `<ASLOT>` 치환 대신, 언어 블록 뒤에 **슬롯당 연속 토큰 1개를 append**(기존 openpi
  "slots" 경로의 검증된 메커니즘 재사용). class-텍스트 사전지식은 프롬프트 단어 + 텍스트에서 시드된
  `audio_class_embedding`으로 보존한다.
- 따라서 프롬프트 텍스트에는 리터럴 `<ASLOT>`를 **넣지 않는다**(치환하지 않으므로 그냥 노이즈 subword가 됨).
  → `InjectAudioBlockText`가 `include_slot_token=False`로 호출.

이 절충으로 학습이 실제로 돌아가는(리스크 낮은) 경로를 확보하면서, 사용자가 고른 핵심 표현
(SlotEncoder+(u,v)+energy, class-as-text)은 그대로 유지된다.

### 3.7 평가 연결 `src/eval/eval_pi05_audio.py`, `src/models/pi05_audio.py`

- `_oracle_snapshot`이 슬롯 az/el을 `(u,v)`로 투영하고 energy를 함께 실어 보냄.
- `_build_policy_batch`가 `observation.audio.energy/uv` 키 추가.
- `AudioAwarePi05Policy`의 `slots_uv` 모드가 energy/uv 컬럼을 openpi 서버로 전송 → 서버가 학습과 **동일한**
  `InjectAudioBlockText`+`BuildAudioSlotsUV`로 프롬프트·슬롯을 만든다(train/eval 파리티의 핵심).

---

## 4. 태스크별 instruction & train/eval 정합

| 태스크 | instruction (train=eval) | 오디오 장면 |
|---|---|---|
| `select_radio` | `primitive: Press the button in front of the radio that is making sound.` | 1개 라디오만 소리 |
| `select_radio_two` | `primitive: Press the button in front of the radio playing the {natural class} sound.` | 2개 라디오, **서로 다른 클래스** |
| `select_radio_silent` | `primitive: Press the button in front of the radio that is silent.` | 3개 중 2개 소리, 조용한 것 선택 |

- eval 기본값이 train과 **같은 `_build_*_instruction`/템플릿**을 사용 → 문자열 완전 일치(검증됨).
  `--revised-instruction`(기본 off)을 켤 때만 "Press"→"Tap" 패러프레이즈로 **의도적** 불일치(일반화 테스트).
- `select_radio_two`는 클래스 이름이 자연어(`_natural_class_name`: `Drums_Percussion`→`drums percussion`)로
  instruction에 들어가며, 오디오 블록은 taxonomy raw명을 쓴다. 표기는 다르나 **train/eval 간엔 동일**하다.

### 4.1 오디오 "값"의 노이즈 정합 (중요)

pi0 학습 데이터는 **clean**(conf=1.0, uv/energy 무노이즈, class=GT)이다. 반면 eval의 오라클 노이즈가 기본
ON이면 train=clean / eval=noisy 미스매치가 난다(SmolVLA는 학습 시 노이즈를 주입해 맞췄지만 openpi 파이프라인은
저장된 clean 값을 그대로 읽는다). → **3-radio 파이프라인 eval을 기본 clean**(모든 노이즈 0, conf 1.0)으로
설정해 정합을 맞춤. 강건성을 보려면 `EVAL_NOISE_AZ_STD=3 ...`로 오버라이드.

> 장기적으로 스펙 §9의 노이즈 커리큘럼(oracle→지터→실 SELD)을 하려면 **학습 시점**에 지터를 넣어야 한다
> (현재는 미구현; 그때는 eval 노이즈도 함께 켜서 맞춘다).

---

## 5. 데이터 균형 (위치 편향)

오라클 생성은 타깃 위치를 `random.choice`로 **균등** 선택하지만, **왼쪽 라디오(radio_0)의 조작 성공률이
구조적으로 낮다**(Franka 워크스페이스 가장자리로 추정). 실패 에피소드는 저장되지 않으므로 저장된 데이터가
middle/right로 쏠린다.

| 태스크 | 위치별 추정 성공률 (균등배치 가정) |
|---|---|
| select_radio | **left 25%** / mid 62% / right 44% |
| select_radio_two | **left 31%** / mid 58% / right 57% |
| select_radio_silent | left 41% / mid 50% / right 54% |

**대응**:
1. `--target-position-label left`로 왼쪽만 top-up 생성(실패는 저장 안 되니 자연히 성공 케이스만 축적).
2. 그래도 남는 불균형은 하향 균등화(과잉 위치를 left 개수에 맞춰 제외; 제외분은 `dataset_seld_uv/_removed_balance_*/`로
   이동해 되돌릴 수 있게 함). 예: `select_radio`를 79/79/79로 맞춤.

> 근본 원인(왼쪽 조작 실패)은 sim/로봇 배치 조정이 필요한 별개 이슈다. top-up은 "성공한 왼쪽 케이스"만
> 모으므로, 왼쪽 실패가 특정 초기배치에 몰려있다면 그 분포까지 교정하진 못한다.

---

## 6. 학습 설정 (frozen vs LoRA)

`OPENPI_PALIGEMMA_LORA=1`(파이프라인 기본, `VLM_LORA=1`) 경로:

| 파트 | 상태 |
|---|---|
| PaliGemma **vision tower (SigLIP)** | **frozen** (LoRA 없음) |
| PaliGemma **LLM (Gemma-2B) base** | **frozen** |
| LLM attention(q/k/v/o_proj)+MLP | **LoRA 학습** (rank 16, α 16, dropout 0.05) |
| action expert, action/state proj | **전체 학습** |
| **SlotEncoder, audio_class_embedding** | **전체 학습** (신규 모듈) |

`VLM_LORA=0`이면 vision tower+LLM까지 전부 학습하는 full fine-tune(설정: `pi0_ft_vlabench_seld_uv_three_radio`).

---

## 7. 실행 (파이프라인)

`sh/train_pi0_seld_uv_three_radio.sh` — 단계 토글(`DO_*`)로 제어.

```
[1] generate  (conda, sim)     : trajectory_generation.py --oracle-mode, 태스크 3종
[2] combine                    : 3태스크 HDF5를 1디렉터리로(one-radio 프롬프트 정규화)
[3] convert   (openpi venv)    : LeRobot v2.1 + energy/uv 컬럼   ← lerobot 0.1.0로 변환해야 함
[4] norm stats (openpi venv)   : compute_norm_stats.py --config-name ...
[5] train     (openpi venv)    : train_pytorch.py, LoRA
[6] eval      (openpi venv)    : eval_pi05_audio.py --audio-mode slots_uv, 3태스크, clean
```

**중요 함정**:
- **변환은 반드시 openpi venv**(lerobot 0.1.0)로: conda(lerobot 0.4.x)는 v3.0(tasks.parquet)을 써서 openpi
  로더(v2.x만)가 HF 404를 낸다. 생성은 sim이 필요해 conda로.
- `compute_norm_stats.py`는 positional이 아니라 **`--config-name`** 플래그.

```bash
export OPENPI_PI0_JAX_WEIGHT=$PWD/checkpoints/pi0_base_primitive/params
export OPENPI_PI0_PYTORCH_WEIGHT=$PWD/checkpoints/pi0_base_primitive_torch
# 재조합→재변환→norm (생성 건너뜀)
DO_GENERATE=0 DO_COMBINE=1 DO_CONVERT=1 DO_NORM=1 DO_TRAIN=0 DO_EVAL=0 bash sh/train_pi0_seld_uv_three_radio.sh
# 학습
DO_GENERATE=0 DO_COMBINE=0 DO_CONVERT=0 DO_NORM=0 DO_TRAIN=1 DO_EVAL=0 bash sh/train_pi0_seld_uv_three_radio.sh
# 평가(clean)
DO_GENERATE=0 DO_COMBINE=0 DO_CONVERT=0 DO_NORM=0 DO_TRAIN=0 DO_EVAL=1 bash sh/train_pi0_seld_uv_three_radio.sh
```

---

## 8. 검증 현황 & 남은 일

**GPU 없이 검증됨**:
- 단위테스트: `tests/test_projection.py`, `tests/test_slot_encoder.py`, `tests/test_slot_serialization.py` 통과.
- openpi transform(`InjectAudioBlockText`/`BuildAudioSlotsUV`) uv venv에서 end-to-end 동작(프롬프트·`[K,6]` 슬롯 확인).
- 신규 TrainConfig 2개 `get_config` 정상 로드.
- 데이터셋 705→(균형)237/294/258 에피소드, intrinsics·energy·uv 컬럼 생성 확인.

**아직 안 함(사용자 GPU 필요)**:
- 실제 pi0 학습 및 3태스크 평가.

**향후**:
- 실 SELD 모델 연결(현재 oracle) — M1 트래커·투영을 추론 루프에.
- 스펙 §9 노이즈 커리큘럼을 **학습 시점**에 주입(그리고 eval 노이즈 동반).
- (선택) inline `<ASLOT>` 토큰 치환 방식(토크나이저 확장) — 현재는 append 토큰.
- (선택) 오디오 블록 클래스어도 자연어로 통일, `max_token_len` 여유 확인.

---

## 부록: 주요 파일

| 파일 | 역할 |
|---|---|
| `src/audio/projection.py` | M2 DoA→(u,v) 투영 |
| `src/audio/slot_encoder.py` | M4 SlotEncoder (canonical) |
| `src/audio/slot_serialization.py` | M3 `<audio>` 블록 직렬화 |
| `src/audio/oracle_sled.py` | 오라클 GT + energy + cam_fovy |
| `src/data/convert_hdf5_to_lerobot.py` | LeRobot 변환 + energy/uv 컬럼 |
| `third_party/openpi/.../pi0_pytorch.py` | SlotEncoder 주입(`_embed_audio_slots`) |
| `third_party/openpi/.../vlabench_policy.py` | `InjectAudioBlockText`, `BuildAudioSlotsUV` |
| `third_party/openpi/.../training/config.py` | `slots_uv` 모드 + TrainConfig 2종 |
| `src/eval/eval_pi05_audio.py`, `src/models/pi05_audio.py` | slots_uv 평가 연결 |
| `sh/train_pi0_seld_uv_three_radio.sh` | 전체 파이프라인 |
| `tests/test_{projection,slot_encoder,slot_serialization}.py` | 단위테스트 |
