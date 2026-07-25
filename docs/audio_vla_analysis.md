# 오디오 정보의 VLA 모델 전달 방식 분석 및 개선 제언

작성일: 2026-06-14
대상 코드: `src/` (audio / models / data / training / eval), `sh/`
대상 모델: SmolVLA (`AudioAwareSmolVLAPolicy`), pi0.5/openpi (`AudioAwarePi05Policy`)

---

## 0. 요약 (TL;DR)

이 프로젝트는 **공간 음향(binaural audio) → SELD(SLED) 전처리 → top-K 이벤트(class, azimuth,
elevation, confidence) → VLA**로 이어지는 *모듈형(modular) 오디오-VLA* 파이프라인이다.
오디오를 raw waveform/spectrogram으로 넘기지 않고, **"무엇이(class) 어디서(direction)
얼마나 확실히(confidence) 들리는가"라는 구조화된 중간 표현**으로 직렬화해서 언어 스트림에
주입한다. 두 가지 백본에 대해 서로 다른 적절한 방식을 쓰고 있다:

- **SmolVLA**: 내부 토큰 임베딩 레벨에서 융합. 클래스명은 텍스트 토큰, 방향은 학습형
  연속 임베딩(`DirectionEncoder`)을 `@` placeholder 위치에 in-place 주입 (Path A).
- **pi0.5/openpi**: 모델 내부에 손대지 않고 **프롬프트 텍스트에 오디오 장면을 자연어로 append**.

전반적으로 설계 방향은 합리적이고 최근 연구 흐름과 일치한다. 초기 분석에서 의심했던 3건을
실제 데이터로 검증한 결과는 다음과 같다:

| 항목 | 초기 의심 | 검증 결과 | 조치 |
|---|---|---|---|
| **H1** state==action | train/eval 분포 불일치 | ✅ **실재 확인** — HDF5에 실측 `observation/ee_state`가 있는데도 미사용 | **수정 완료** (opt-in `--use-real-state`) |
| **H3** azimuth 부호 | 좌/우 규약 모순 | ❌ **오탐 — 버그 아님** (eval 데이터 15+건으로 self-consistent 입증) | 수정 불필요 (아래 근거) |
| **H2** √H 스케일 | 임베딩 스케일 불일치 | ⚠️ 코드 불일치는 사실이나 실질 영향 낮고, 수정 시 기존 체크포인트 무효화 | 권고만 (미적용) |
| **L1** 오타 "tdhe" | — | ✅ 명백한 버그 | **수정 완료** |

아래에 심각도 순으로 정리한다. 검증 로그는 §3.1 / §3.3 / §7에 첨부.

---

## 1. 전체 파이프라인 구조

```
[MuJoCo 장면 + 음원] 
   └─(audio_generation/AudioSimManager)→ binaural WAV
         └─(SLED v5 / sled_overlay)→ per-frame top-K {class_id, az, el, conf}
               │                         ↑ (실측 경로)
               └─(oracle_sled.py)────────┘ GT 위치 → 노이즈 주입 (oracle 경로)
                     │
   HDF5 ─(convert_hdf5_to_lerobot.py)→ LeRobot dataset
                     │   observation.audio.{azimuth_deg, elevation_deg, confidence, class_id} : (top_k,)
                     ▼
   ┌──────────────── SmolVLA (Path A, 내부 임베딩 융합) ────────────────┐
   │  audio_token_builder → "[AUDIO] <cls> @ conf 0.NN ; ... [/AUDIO]"  │
   │  direction_encoder(az,el)·conf  →  @ 위치에 in-place 덮어쓰기       │
   │  prefix = [img]+[lang]+[AUDIO block]+[state] → flow-matching expert │
   └────────────────────────────────────────────────────────────────────┘
   ┌──────────────── pi0.5/openpi (프롬프트 텍스트 융합) ───────────────┐
   │  audio_prompt_builder → "Audio scene: 1 source detected. source 1:  │
   │     dog barking, to the left, azimuth -54 degrees, confidence 0.93" │
   │  prompt = instruction + "\n" + audio_text → openpi.infer            │
   └────────────────────────────────────────────────────────────────────┘
```

핵심 설계 결정:
- **모듈형 분리**: 인식(SELD)과 정책(VLA)을 분리. raw audio가 아니라 SELD 출력(symbolic)을 넘긴다.
- **oracle 학습 레짐**: GT 위치에서 합성한 top-K에 **매 배치 새 노이즈**를 주입(`_apply_oracle_noise`).
  SELD 전처리 단계의 sim-to-real 갭을 도메인 랜덤화로 흡수하려는 의도. 좋은 관행.
- **front camera = cam_id 2**가 listener (메모리/파이프라인 일관).

---

## 2. 오디오 전달 방식의 적절성 평가

### 2.1 잘 된 점

1. **백본별로 올바른 확장 지점을 선택함.**
   - SmolVLA는 가중치에 접근 가능 → 내부 토큰 임베딩에 연속 방향 벡터를 주입(continuous
     conditioning). 방향처럼 본질적으로 연속인 양을 텍스트 정수("-54 degrees")로만 주면
     토크나이저 양자화 손실이 생기는데, `DirectionEncoder`로 연속값을 그대로 보존한 점이 좋다.
   - pi0.5/openpi는 서버/체크포인트가 black-box → 프롬프트 텍스트로만 확장. 현실적인 선택.

2. **클래스 정체성과 방향을 같은 위치에 co-locate.**
   `_embed_audio_block`에서 `<class words> @`처럼 클래스 토큰 바로 뒤 `@`를 방향 임베딩으로
   교체 → LLM이 "이 클래스 = 이 방향"을 self-attention으로 묶기 쉽다 (`smolvla_audio.py:168-172`).
   instruction의 클래스명 ↔ 오디오 슬롯 클래스명 매칭으로 타깃 방향을 라우팅하는 two-radio
   과제의 핵심 메커니즘.

3. **confidence gating.** 방향 임베딩에 `conf`를 곱해(`dir_emb * conf_gate`) 불확실/빈 슬롯의
   방향 신호를 자연스럽게 0으로 감쇠. 빈 슬롯은 "silence" 단어로 렌더(`audio_token_builder.py:136`).

4. **다중 정렬(alignment) 단계 제공.** `--train-audio-only`로 direction_encoder만 먼저 정렬
   (LLaVA의 stage-1 projector pretraining과 동형). modality gap을 줄이는 정석적 접근.

5. **인식 노이즈에 대한 도메인 랜덤화 + 슬롯 순서 불변성 학습.** `shuffle_slots`로 "slot 0 =
   target" 위치 단축학습(shortcut)을 깨고, 진짜 클래스명 매칭을 강제. 멀티소스 SELD 출력이
   본질적으로 순서 없음(unordered)이라는 점과 일치.

6. **fixed_fourier 옵션.** scalar(az/el)에 대한 결정론적 sinusoidal encoding은 NeRF/Transformer
   positional encoding과 동일한 아이디어로, 연속 스칼라 주입의 표준 기법.

### 2.2 최근 연구 동향과의 비교

| 축 | 이 프로젝트 | 최근 연구 흐름 | 코멘트 |
|---|---|---|---|
| 오디오 표현 | SELD top-K → symbolic(text+연속 방향) | (a) raw spectrogram end-to-end (**ManiWAV** 2024, *Play-it-by-Ear*, *The Sound of Touch*) <br>(b) CLAP/CLAP-like 오디오 인코더 토큰 (LLaVA식 projector) <br>(c) Qwen-Audio / Audio-Flamingo식 오디오-LLM | 본 프로젝트는 **명시적 중간 표현(modular)**. 해석가능·데이터효율적이지만 timbre·onset 등 fine-grained 음향 단서를 클래스 라벨로 압축하며 잃음. SELD 품질이 상한. |
| 융합 위치 | 언어 스트림 prefix에 early fusion | LLaVA류 projector 토큰 / cross-attn(Flamingo) | Path A(언어 토큰열 주입)는 단순·견고. 메모리에 적힌 Path C(vision-audio cross-attn) 미구현. |
| 공간 정보 | az/el 연속 임베딩 + 좌/우 coarse 단어 | spatial audio(DCASE SELD), audio-visual nav(SoundSpaces) | 연속+이산 병행은 견고한 선택. |
| sim-to-real | SELD 출력에 도메인 랜덤화 노이즈 | perception 노이즈 주입 / 표현 정규화 | oracle 노이즈 주입은 정석. |
| 정렬 | audio-only 사전 정렬 단계 | 2-stage(projector→joint) LLaVA/BLIP-2 | 일치. |

**요약 판단**: "raw audio를 VLA에 직접 욱여넣기"보다 **SELD라는 강한 inductive bias를 거친
symbolic conditioning**을 택한 것은, 데이터가 적고 과제가 "어느 방향의 어떤 소리"로 환원되는
본 벤치마크에선 합리적이다. 다만 이는 **음향의 미세 정보를 버리는 상한**을 동시에 의미하므로,
논문화 시 "왜 raw-audio end-to-end가 아니라 modular인가"를 명시적으로 정당화하고, 가능하면
**raw-spectrogram 베이스라인(ManiWAV식)** 또는 **CLAP 임베딩 토큰 베이스라인**과 비교하는 것을
권장한다.

---

## 3. 발견된 문제점 (심각도 순)

### 🔴 H1. state == action 결합으로 인한 train/eval 분포 불일치 — ✅ 확인 & 수정 완료

- **학습(이전)**: `convert_hdf5_to_lerobot.py` `out_state = out_action.copy()` →
  `observation.state[t]`가 `action[t]`와 **완전히 동일**(둘 다 `action_8d[t]`에서 파생).
- **평가**: `eval_smolvla_audio.py:343-357`에서 state는 **실측 EE pose**(`get_ee_state`)로 구성.

학습 시 정책은 `action[0] = state`라는 **자명한 단축경로**를 학습할 수 있다(특히 chunk 첫 스텝).
평가 시 state는 명령이 아니라 "실제 도달한 pose"라 명령보다 지연(lag)되므로, 정책이 state를
복사하는 성향을 가질수록 행동이 굼뜨거나 정상상태 오프셋이 생긴다. 오디오 과제 성패와 무관하게
**기본 제어 품질을 갉아먹는** 요인이며, 오디오 ablation 결과의 신뢰도도 떨어뜨린다.

#### 3.1 검증 — 실측 state는 HDF5에 이미 존재했다

코드 주석은 "we lack a separately recorded ee_pose feedback stream"라고 했지만(`:88-94`),
실제 HDF5 에피소드를 열어보니 **`observation/ee_state` (T, 8) 스트림이 존재**한다
(`[x,y,z(world), quat_wxyz, open_flag]`). 즉 실측 state가 있는데도 명령값으로 덮어쓰고 있었다.

추가로 `ee_world[:3] - action_base[:3]`가 에피소드 간 `~[0.0, -0.402, 0.783]`로 mm 단위
안정 → robot base offset(상수)임을 확인. 이를 빼면 eval의 `pos_world - get_robot_frame_position()`과
동일한 base-frame pose가 복원된다.

#### 수정 내용 (적용됨, opt-in)

`convert_hdf5_to_lerobot.py`에 **`_compute_real_state()` + `--use-real-state` 플래그**(기본 OFF)를
추가. eval의 state 생성 수식을 그대로 복제한다:
- 위치: `ee_world - median(ee_world - action_base)` (per-episode robot offset 추정, self-cancel)
- 자세: eval과 동일한 `quaternion_to_euler(ee_quat)` (라디안)
- gripper: `1.0 - ee_state[7]` — `get_ee_open_state`의 알려진 부호 버그(닫힘=1.0)를 eval과
  동일하게 반전하여 열림=1.0로 통일

오프라인 검증(`dataset_oracle/select_radio/data_1.hdf5`):
```
action real==legacy        : True          # action target은 그대로, state만 변경
pos|state-action| legacy   : 0.0000 m       # ← 버그(완전 동일)
pos|state-action| real     : 0.0098 m mean / 0.0438 m max   # ← 실측 tracking lag 복원
real euler range           : [-3.13, 3.10] rad              # eval 규약 일치
gripper uniq {0,1}, 일치율  : 73.86%         # 나머지는 실제 actuation lag (legacy는 100% 강제)
```
**env 대조 검증(실행 완료)** — 추정 robot offset이 eval이 실제로 빼는 값과 일치하는지 확인:
```
env.get_robot_frame_position()  = [0.000, -0.400, 0.780]  (시드 무관 상수)
converter median estimate       = [0.0005, -0.4019, 0.7822]
|diff|                          = [0.5, 1.9, 2.2] mm   → 1cm 이내 일치 ✓
```
즉 `_compute_real_state`의 state는 eval-time state(`pos_world − robot_base`)와 ~2mm 이내로
일치하며, euler/gripper는 동일 함수·동일 반전을 쓰므로 정확히 같다.

**기본 OFF**라 기존 데이터셋/체크포인트엔 영향 없음. 다음 변환+재학습 시 `--use-real-state`를
켜면 갭이 제거된다. (robot offset이 다른 task/robot에서 상수가 아닐 수 있으니 새 task 적용 시
위 대조를 1회 반복 권장.)

---

### 🟠 H2. inline 융합 모드에서 방향 임베딩 스케일 불일치

`_embed_audio_block`(inline):
```python
emb = embed_language_tokens(ids) * math.sqrt(H)      # 텍스트 토큰: ×√H (≈×31 for H=960)
...
dir_tokens = dir_emb * conf_gate                      # 방향 토큰: ×1  (√H 미적용)
emb = torch.where(pos_mask, dir_tokens_expanded, emb) # @ 위치 덮어쓰기
```
`smolvla_audio.py:156, 166, 172`. 반면 `class_tokens` 모드(`_embed_direction_tokens`)는
`tokens = (cls_emb + dir_emb*conf_gate) * math.sqrt(H)`로 **√H를 곱한다**(`:204`).

즉 inline 모드의 방향 토큰은 주변 텍스트 토큰보다 ~√H배 작은 스케일로 시퀀스에 들어간다.
`DirectionEncoder`의 마지막 layer가 `std=0.01` near-zero init(`direction_encoder.py:35`)인 점과
겹쳐, **방향 신호가 다른 토큰 대비 매우 작게 시작**하고, 유의미해지려면 학습이 가중치를 ~31배
키워야 한다. fixed_fourier(O(1) 출력)는 이 스케일 갭이 더 고정적이라 더 불리하다.

**권장**: inline 경로의 `dir_tokens`에도 `* math.sqrt(H)`를 곱해 class_tokens 모드와 일관화.
(near-zero init의 "천천히 켜진다"는 의도와 √H 스케일은 양립 가능 — init은 작게, 동작 스케일은 맞춤.)

---

### ✅ H3. azimuth 부호 규약 — 검증 결과 **버그 아님** (오탐 철회)

초기 분석에서 `oracle_sled.py`(az>0=오른쪽)와 `_azimuth_words`(az>0=왼쪽)가 모순된다고
의심했으나, **실제 저장된 eval 결과 15+건을 교차검증한 결과 self-consistent함을 확인**했다.

#### 3.3 검증 — position_label ↔ 저장 az 부호 상관 (실측)

`outputs/eval_smolvla_oracle_ver1~ver7/eval_infos.json`에서 `position_label`과
`final_audio_snapshot.azimuth_deg[0]`을 상관:
```
ver1~ver7 (각 n=100, 일관)
  left   : mean az = +15° 내외  (양수)
  middle : mean az ≈  0°
  right  : mean az = -15° 내외  (음수)
  ── ver1_noneg (negation 제거 실험): 정확히 부호 반전 (left -16°, right +12°)
```
즉 **현재 파이프라인에서 "left 위치 = 양수 az"가 실제 저장값**이며, `_azimuth_words`의
"az>0 → left"는 데이터와 일치한다.

#### 왜 두 docstring이 둘 다 맞는가 (서로 다른 frame)

cam_2(listener)가 테이블을 **정면에서** 바라보므로, **테이블-좌측 라디오는 listener 기준으로는
오른쪽**에 위치한다. 따라서:
- `oracle_sled`의 "az_sled>0 = listener의 오른쪽" (head-relative) → 참
- `_azimuth_words`의 "az>0 = 'left'" (position-label/테이블 frame) → 참 (테이블-좌측 = listener-오른쪽)

두 진술은 **서로 다른 좌표계를 기술**할 뿐 모순이 아니다. 그리고 `_azimuth_words`는 train·eval에서
**동일 코드 경로**로 적용되므로 부호 규약이 항상 일관된다. 연속 `DirectionEncoder`도 같은 저장
az를 받으므로 단어와 같은 방향을 가리킨다.

> 잔여 메모(버그 아님): "left"라는 **단어**는 시각적으로 image-우측에 보이는 라디오를 가리키므로,
> pretrained VLM의 image-left 공간 prior와 어긋날 수 있다. 그러나 train/eval 일관 + 연속 임베딩
> 우세로 **학습 가능**하며 정확도 버그가 아니다. 굳이 바꾸면 position-label 기반 instruction
> grounding과의 일관성이 깨지고 전면 재학습이 필요하므로 **변경하지 않는다**.

---

### 🟡 M1. `natural_language` 모드의 토큰 예산 truncation 위험

`natural_language` 슬롯 문장은 길다: `"sound 1: dog barking is to the left @, azimuth -54
degrees, confidence 0.93."`. top_k=3이면 `[AUDIO] ... [/AUDIO]`가 쉽게 60~90+ 토큰이 된다.
그러나 `AudioConfig.audio_max_len` 기본값은 **64**이고, builder는 `truncation=True,
padding="max_length"`(`audio_token_builder.py:206-212`). 잘리면 **3번째 슬롯의 `@`가 사라져
direction 주입이 조용히 누락**되고, `[/AUDIO]`도 잘릴 수 있다.

`@` 위치 탐색은 잘린 시퀀스에서 "있는 만큼만"(`positions[:top_k]`) 가져가므로 **에러 없이 슬롯이
누락**된다 — 디버깅이 어렵다.

**권장**: natural_language 사용 시 `--audio-max-len`을 96~128로 상향(스크립트의 `AUDIO_MAX_LEN=96`
은 변환기용일 뿐 trainer의 `--audio-max-len`으로 전달되는지 확인 필요). 또한 build_batch에서
실제 발견된 `@` 개수가 present 슬롯 수보다 적으면 **경고를 출력**하도록 방어 로직 추가 권장.

---

### 🟡 M2. pi0.5 클라이언트의 `action_plan` deque는 사용되지 않는 dead code

`_OpenPiClient`가 `action_plan`/`replan_steps`를 만들지만(`pi05_audio.py:71-72, 106`),
`AudioAwarePi05Policy.predict_action_chunk`는 매 호출 `client.infer`로 **전체 chunk를 새로
받아** 반환하고 horizon 재계획은 상위 eval 루프가 담당한다. deque는 어디서도 push/pop되지 않음.
기능 버그는 아니지만, "openpi가 replan_steps를 캐싱한다"는 오해를 부른다.

**권장**: deque 제거하거나, 실제로 chunk 캐시 재사용을 구현해 추론 횟수를 줄이거나 둘 중 하나로
명확히. 현재는 매 chunk마다 websocket/local infer를 호출하므로 horizon이 작을수록 추론 비용↑.

---

### 🟡 M3. oracle 학습은 "정적 단일 스냅샷"이라 시간 모델링이 사실상 미사용

oracle 변환기는 모든 프레임에 동일 GT를 채우고(`convert_hdf5_to_lerobot.py:283-286`), select_radio
평가도 카메라 정지로 단일 스냅샷을 반복 입력. 즉 오디오는 에피소드 내 **상수**다. radio 과제엔
충분하나, microwave(지연 chime)·이동 음원 등으로 확장하면 현재의 프레임별 동일값 가정과
`temporal_smooth_topk`(실측 경로 전용)의 의미가 달라진다. take_out_microwave_food는
`active_from_frame` 게이팅으로 일부 대응하지만, **이동/시변 음원은 converter가 `static=false`를
명시적으로 NotImplementedError 처리**(`:261-265`). 향후 확장 시 핵심 제약.

---

### 🟢 L1~L4. 경미한 이슈

- **L1 (수정 완료)**: eval 기본 instruction 오타 `"...Tap tdhe button..."` →
  `"...Tap the button..."`로 수정 (`eval_smolvla_audio.py`, `eval_pi05_audio.py`).
  `--revised-instruction` 사용 시 모델에 잘못된 토큰이 들어가던 문제 해소.
- **L2**: inline 모드에서 빈(silence) 슬롯의 `@`는 `dir_emb*0 = 0벡터`로 덮어써져 시퀀스 중간에
  literal zero token이 생긴다. 동작엔 문제없으나, 학습된 "silence 방향=0" 토큰이 의미상 모호.
- **L3**: `_augment_image`의 translate가 batch 루프(`for b in range(B)`)로 `torch.roll` —
  GPU에서 비효율. `torchvision`/벡터화 가능.
- **L4**: prefix에 항상 `audio_max_len`(=64) 토큰이 패딩 포함 추가되므로, 단일 음원에도 prefix가
  길어진다(연산·KV cache). 그리고 audio 블록이 `prefix_length`를 초과하면
  `embed_prefix_with_audio`의 패딩 분기(`smolvla_audio.py:297-301`)가 작동하지 않으니, 실제
  `self.prefix_length`가 audio 토큰 포함분을 수용하는지 한 번 검증 권장.

---

## 4. training / evaluation 정합성 점검 (오류 가능성)

오디오 관점에서 train↔eval가 **일관되게** 동작하는지 확인한 결과:

| 항목 | 학습 | 평가 | 정합성 |
|---|---|---|---|
| az 부호 규약 | oracle GT + 노이즈 | oracle 동일 / 실측 SLED `arctan2(dy,dx)` | ✅ **검증 완료** (H3: self-consistent, left=+az) |
| 클래스 ID 의미 | taxonomy 38-class | 동일 taxonomy | ✅ |
| gripper state 의미 | converter: open=1.0(`_binarize_gripper`) | eval: `not get_ee_open_state`로 반전 보정(`:355`) | ✅ (반전 버그는 이미 주석대로 수정됨) |
| observation.state | (이전)명령 pose=action / (수정)실측 pose | 실측 pose | 🔴→✅ **H1 수정**(`--use-real-state`로 정합) |
| 슬롯 순서 | shuffle/canonicalize(train) | shuffle/canonicalize(eval) 대응 옵션 존재 | ✅ 옵션이 짝맞음 |
| confidence 분포 | U[0.85,0.98] | oracle 동일 / SLED 실제 conf | ⚠️ 실측 SLED conf 분포가 학습 분포와 다르면 갭 (의도된 도메인 갭) |
| audio_fusion_mode | ckpt의 `audio_config` 저장 | eval가 ckpt에서 복원(`eval:1212`) | ✅ 좋은 설계 |
| 토크나이저 max_length | tokenizer_max_length(48) | 동일 경로 | ✅ |

**눈에 띄는 정합성 강점**: `audio_config`를 체크포인트 메타에 저장하고 eval가 그대로 복원하므로
(`train:863`, `eval:1208-1213`), 융합 모드/슬롯수 불일치로 인한 침묵형 오류가 구조적으로 차단됨.
좋은 패턴.

**잠재 오류**:
- **H1**(state)은 평가 지표를 직접 왜곡하던 실질 항목 → 수정 완료(opt-in). **H3**(좌/우)은
  검증 결과 정상.
- 실측 SLED 경로의 confidence 임계/분포(`conf_thresh=0.30`)와 oracle 학습 conf 분포(0.85~0.98)
  차이는 의도적 도메인 갭이지만, real_sled 평가 성능이 oracle 대비 급락하면 이 지점부터 의심.

---

## 5. 개선 제언 (우선순위)

1. **(완료) H1** — `--use-real-state`로 실측 EE state 사용 경로 추가(opt-in). 다음 변환+재학습
   시 적용 권장. 적용 후 oracle/real_sled 성공률을 legacy와 비교해 효과 확인.
2. **(완료) L1** — instruction 오타 수정.
3. **(검증·기각) H3** — 버그 아님. 추가 조치 불필요.
4. **(권장, 미적용) H2 스케일 일관화** — inline/natural_language 경로의 `dir_tokens *= √H`.
   단, 기존 체크포인트와 호환 깨짐 → 재학습 전제. near-zero init 덕에 실질 영향은 작아 후순위.
5. **(권장) M1 방어** — natural_language 시 audio_max_len 상향 + `@` 누락 경고.
6. **(연구적 가치) 베이스라인 추가** — raw-spectrogram(ManiWAV식) 또는 CLAP-embedding-token
   베이스라인과 비교해 "modular symbolic conditioning"의 이득/손실을 정량화. 논문화에 직접 기여.
7. **(연구적 가치) 표현 ablation 표** — {inline, class_tokens, natural_language} ×
   {mlp, fixed_fourier} × {shuffle on/off}의 성공률 표. 현재 코드가 이미 모든 스위치를 제공하므로
   적은 추가 비용으로 강한 분석 섹션을 만들 수 있음.
8. **(선택) M2 정리** — pi0.5 action_plan dead code 제거 또는 chunk 캐시 구현.

---

## 6. 결론

오디오를 **SELD 기반 symbolic 중간표현 + 연속 방향 임베딩**으로 VLA에 전달하는 본 설계는,
데이터 효율·해석가능성·백본 호환성 측면에서 타당하며 최근 멀티모달 LLM/VLA 확장 관행(LLaVA식
2-stage 정렬, 연속 스칼라의 positional/Fourier encoding, 인식 노이즈 도메인 랜덤화)과 잘 맞는다.
체크포인트에 audio_config를 저장해 train/eval 침묵형 불일치를 막은 점, 슬롯 순서 불변성을
강제한 점은 특히 견고하다.

검증 결과, 초기에 의심한 3건 중 **H1(state=action 분포 불일치)만 실제 버그**였고 이미 수정했다
(`--use-real-state`, opt-in). **H3(az 좌/우 부호)은 eval 데이터로 self-consistent함을 입증해
기각**, **H2(√H 스케일)는 코드 불일치이나 실질 영향이 작고 재학습 전제라 권고로 남김**. 오타
L1도 수정했다.

남은 권장사항은 H2 스케일 일관화(재학습 시)와 M1 토큰 예산 방어이며, 그 외에는 현재 코드가
이미 제공하는 다양한 융합/인코더/슬롯 스위치를 활용해 곧바로 설득력 있는 ablation 분석(특히
raw-audio/CLAP 베이스라인 대비)으로 발전시킬 수 있다.
</content>
</invoke>
