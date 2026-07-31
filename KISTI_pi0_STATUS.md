# find_hidden / SELD-VLA pi0 — 진행상황 핸드오프

> 마지막 업데이트: 2026-07-31 (v3가 감사를 통과하는데도 성능이 낮았던 이유: roll이 오일러 ±π 경계선 위 — 맨 아래 절 참고)

---

## 🔴 한 줄 요약

find_hidden이 계속 **0% 성공률**이던 원인은 **학습 데이터와 평가의 로봇 base frame 원점이 28.8cm 어긋나 있던 것**이었다.
변환기를 고쳐 재학습하니 **처음으로 0을 벗어났다 (5/100)**. 남은 병목은 **고도각(위/아래 서랍) 단서**다.

---

## 🐞 근본 원인 — base frame 원점 불일치

### 측정 (2026-07-29, 시뮬레이터에서 직접 읽음)

같은 물리적 로봇 자세(월드 `[0.0004, -0.4703, 1.1319]`, 에피소드 첫 프레임)를 정책이 이렇게 다르게 받고 있었다:

```
학습 데이터 :  x -0.0017   y -0.0580   z +0.3574
평가 시뮬   :  x +0.0004   y +0.2297   z +0.4319
차이        :        +0.002      +0.288      +0.074      → |Δ| = 0.296 m
```

| | `get_robot_frame_position()` (평가가 쓰는 값) | 기록된 action의 프레임 원점 | 차이 |
|---|---|---|---|
| **find_hidden_object_open** | `(0, -0.700, 0.700)` | `(0.0021, -0.4123, 0.7744)` | **(0.002, +0.288, +0.074)** |
| **select_radio** | `(0, -0.400, 0.780)` | `(0.0002, -0.4024, 0.7819)` | (0.0002, -0.002, +0.002) ≈ 0 |

**왜 find_hidden만 어긋났나**: find_hidden 씬은 캐비닛 두 개를 놓기 위해 로봇 base를 뒤로·아래로 옮겼다
(`y -0.4 → -0.7`, `z 0.78 → 0.70`). 그런데 궤적 생성기는 **기본 위치 기준**으로 action을 기록한다.
radio는 로봇이 기본 위치 그대로라 애초에 어긋날 일이 없었다 — **radio에는 이 버그가 없다.**

### 왜 치명적이었나

평가의 `eval_smolvla_audio._apply_action`은 `pos_world = pos_base + get_robot_frame_position()`으로 되돌린다.
즉 **state 입력과 action 출력 양쪽 모두** 이 프레임을 쓴다. 결과적으로 정책은 t=0에 "y가 이미 +0.23"이라고
본다. 학습 궤적에서 y=+0.23은 **도달의 약 80% 지점**이라, 정책은 "거의 다 왔다"고 판단하고 조금만 더
가다 멈춘다. 관측된 "4~5초 부드럽게 움직이다 정지 후 버벅거림"이 정확히 이것이다.

### 증상이 축별 오차와 정확히 비례했다

| 축 | 프레임 오차 | 관측된 정책 오차 |
|---|---|---|
| x | +0.002 m | 달성률 69% (거의 정상) |
| z | +0.074 m | 움직이면 안 되는데 8.5cm 하강 |
| y | +0.288 m | 전진 24.4cm 부족 |

### 배제 과정에서 확인된 것 (전부 원인이 아니었음)

| 가설 | 반증 |
|---|---|
| 카메라 위치 | 새 카메라로 880ep 재생성 → 여전히 0% |
| 데이터 부족 / 학습 부족 | 880ep·20k, KISTI 4장 20에폭 → 전부 0% |
| `state == action` 데이터 결함 | real-state 재변환 후에도 0% (별개의 실재 결함이긴 함) |
| 고도각 단서가 약해서 | top-only(440ep) 학습도 0% ← *단, 이 실험은 프레임 버그에 가려져 무효였음* |
| 재계획 폐루프 드리프트 | `horizon=50` 개루프도 동일 (오히려 더 나쁨) |
| IK / 추종 실패 | 최종 추종오차 1.2mm, IK 실패 0건 |
| 물리적 간섭 | 명령한 자세에 정확히 도달함 |
| 깊이(depth) 관측 불가 | teacher-forced로 반증: 학습 관측에선 y를 96.6% 정확도·상관 0.998로 예측 |

**결정적 실험이었던 teacher-forced 검사**: 학습 데이터 관측을 그대로 넣으면 모델은 전진을 정확히 예측한다.
같은 모델이 평가 관측에서는 +0.001m만 전진한다 → 모델이 아니라 **관측이 다르다**는 결론.

---

## ✅ 수정

`src/data/convert_hdf5_to_lerobot.py`:

```python
ROBOT_FRAME_POS = np.array([0.0, -0.7, 0.7])   # find_hidden 씬의 get_robot_frame_position()

action_frame_origin = np.median(ee_pos_world - out_action[:n, :3], axis=0)
shift = action_frame_origin - ROBOT_FRAME_POS
out_action[:, :3] += shift                        # action을 평가 프레임으로
out_state[:, :3]   = ee_pos_world - ROBOT_FRAME_POS  # state도 평가 프레임으로
```

**state와 action을 같이 옮겨야 한다.** 한쪽만 고치면 서로 어긋나 더 나빠진다.

검증 (재변환 전):
```
수정 후 첫 프레임 state  [0.0005  0.2290  0.4301]
평가 시뮬 실측           [0.0004  0.2297  0.4319]      오차 y -0.7mm, z -1.8mm  ✅
y 전진량 0.362 m — 수정 전과 동일 (순수 평행이동이라 궤적 자체는 안 변함)  ✅
```

⚠️ `ROBOT_FRAME_POS`는 **find_hidden 씬 상수를 하드코딩**한 것이다. 다른 태스크를 변환할 땐 그 태스크의
`get_robot_frame_position()`을 확인해야 한다 (radio는 `(0,-0.4,0.78)`이고 이미 일치하므로 손대면 안 됨).

---

## 📊 결과

### framefix 재학습 (로컬, 2026-07-28~29)

880ep · batch 16 · 20,000 steps · LoRA r16 · real-state + frame fix

```
step 20000, 100 에피소드
success 0.050   intention 0.080   progress 0.0350      (이전 전부 0.000)

  right_top     3/24  = 0.125          left_top      2/35  = 0.057
  right_bottom  0/19  = 0.000          left_bottom   0/22  = 0.000
```

도달 회복 확인:
```
종료 EE y (평가 base frame, 데모 목표 +0.59)
  left_top +0.522   right_top +0.501   left_bottom +0.405   right_bottom +0.428
  이전(버그): 같은 좌표계로 +0.348 에서 정지
소진 스텝 244.1/250 (이전 250.0 = 전부 상한 소진, 조기 성공 0건)
```

### ~~다음 병목: 고도각~~ → **철회됨 (2026-07-30)**

> 아래 원문은 **틀렸다.** 2026-07-30 실험 3건으로 반증됐다. 기록용으로 남긴다.
>
> ~~성공 5건이 전부 top 슬롯, bottom은 0/41. 방위각 분리/잡음 5.8 ✅ vs 고도각 1.8 ❌.
> 고도각은 두 분포가 거의 겹친다. **모델이 못 배운 게 아니라 구별할 정보가 없다.**
> 선택지: 서랍 수직 간격 확대 / 마이크 재배치 / `src/audio_visual_bridge`.~~

**반증 1 — 고도각 정보는 충분하다.** 정책이 실제로 받는 `slots_uv` 피처(880ep)로 프로브를 돌리면:

| 피처 | top | bottom | d′ |
|---|---|---|---|
| `v` (투영 세로좌표) | 0.418 | 0.594 | **6.41** |
| `el_deg` | +4.25° | −4.90° | **6.43** |
| `u` (가로) | 0.499 | 0.498 | 0.00 |

**스칼라 `v` 하나만으로 로지스틱 회귀가 top/bottom 을 99.5% (5-fold CV) 로 분류한다.**
에피소드 내 `v` 분산은 0.0000 — 오라클 피처라 프레임 잡음이 아예 없다. 방위각은 `(u,v)` 로 100%.
위 “분리/잡음 1.8” 은 다른 조건(구 카메라 등)에서 잰 값으로 보인다. 현재 데이터엔 해당 없음.

**반증 2 — bottom 전용 학습이 전혀 도움이 안 된다.** bottom 440ep 만으로 학습
(`pi0_ft_vlabench_find_hidden_bottom_lora`, 4슬롯 런과 동일 조건, loss **0.0049** 로 수렴 — 4슬롯의 0.008보다 낮다):

| 정책 | bottom 100ep 성공 | 최종 y (데모 +0.61) | 최종 z (데모 +0.161) |
|---|---|---|---|
| bottom 전용 | **0/100** | +0.383 | +0.209 |
| 4슬롯 (0.050) | **3/100** | +0.414 | +0.237 |

전용 학습이 오히려 미세하게 나쁘다 (Fisher p=0.25, 즉 차이 없음). 데이터 비율·슬롯 경쟁 문제가 아니다.

**반증 3 — “bottom 0” 자체가 표본 오류였다.** 위 0/41 은 참값 3% 일 때 관측 확률 **0.29** 로 흔한 일이다.
n=100 으로 다시 재니 bottom 은 0 이 아니라 **3%**. top 5/59(8.5%) 와 비교해도 **p=0.148 — 유의하지 않다.**

### 그럼 진짜 병목은: **정지(stall)**

두 정책 모두 손잡이 앞 **0.11~0.20 m 지점에서 멈춘다** (top 0.52 vs 0.63, bottom 0.41 vs 0.61).
250스텝을 전부 소진하고, `progress` 는 정확히 0.000 — 서랍을 아예 건드리지 못한다.
그런데 `ik_fail_rate 0.000`, 최종 추종오차 3.4 mm — 컨트롤러는 시킨 대로 정확히 움직인다.
**즉 정책이 “더 가라”는 명령을 스스로 멈추는 것이다.** 프레임 버그 시절의 “부드럽게 가다 정지” 와
같은 신호이며, frame fix 는 정지 지점을 +0.348 → +0.4~0.5 로 옮겼을 뿐 없애지 못했다.

성공률이 top 8.5% / bottom 3% 로 **어디서나 낮은** 것도 같은 원인으로 설명된다.
오디오 쪽 작업(서랍 간격·마이크 배치·`audio_visual_bridge`)은 이 숫자를 못 움직인다.

다음 진단: **teacher-forced 비교** — 학습 관측 vs 평가 관측을 같은 자세에서 넣어 예측 청크를 비교한다
(프레임 버그를 잡았던 바로 그 검사). 학습 관측에선 정확한데 평가 관측에서 delta 가 0 으로 붕괴하면
관측 불일치가 하나 더 남아 있는 것이고, 데모 매니폴드에서 멀어질수록 서서히 나빠지면 통상적인
compounding error 라 처방이 다르다(horizon·재계획 주기·DAgger).

### KISTI staged-freeze (2026-07-28~29)

| Job | 태스크 | 전환 | steps·batch | 소요 | 성공률 | baseline |
|---|---|---|---|---|---|---|
| 865268 | find_hidden | step 8400/14000 (**60%**) | 14k · 16 | 18h55m | **0.000** | 버그 데이터 → 무효 |
| 865340 | select_radio | step 14000/20000 (**70%**) | 20k · 16 | 14h20m | **0.810** | 로컬 0.690 |

전환 로그 (두 잡 동일):
```
[staged-freeze] step N: PaliGemma frozen (2942.9M params, 19.6M newly frozen);
                trainable now 578.7M (action expert + heads)
[staged-freeze] carried AdamW state for 181/182 surviving tensors
```
`19.6M newly frozen` = LoRA 어댑터. **어댑터까지 함께 동결**하는 의도대로 동작.

#### ✅ 865340 (radio, 70%) — staged-freeze 를 지지하는 **유일한 유효 실험**

`outputs/eval_radio_kisti_stagedfreeze/` · step 20000 · 100 에피소드 · oracle · slots_uv

```
success 0.810   intention 0.900   progress 0.810        (로컬 baseline 0.690)

  left   27/35 = 0.771      middle 25/33 = 0.758      right 29/32 = 0.906
  intention: left 0.829     middle 0.909              right 0.969
```

**0.690 → 0.810.** radio 데이터엔 프레임 버그가 없으므로 로컬 baseline 과 직접 비교되는 유효한 결과다.
이후 모든 런(v3 pi0.5 로컬, KISTI 867002/867003)을 **70%로 맞춘 근거가 이 실험 하나**다.

⚠️ **다만 근거 강도를 과대평가하지 말 것.** 표본은 태스크 1개·실험 1회이고, baseline 과 batch·steps 가
같아야 staged-freeze 단독 효과라고 말할 수 있다. 지금 진행 중인 런들은 batch 가 다르다
(로컬 v3 = 24, KISTI = 64) → 그 결과로 70% 의 우수성을 다시 주장할 수는 없다.

#### ❌ 865268 (find_hidden, 60%) — 무효

`outputs/eval_find_hidden_kisti_stagedfreeze/` · step 14000 · 100 에피소드 → **success 0.000**,
슬롯별 전부 0 (left_top 0/35, right_top 0/24, left_bottom 0/22, right_bottom 0/19).

프레임 버그가 있는 데이터로 학습됐으므로 framefix(0.050)와 직접 비교 불가.
비교 대상은 같은 버그 데이터로 학습한 이전 것들(전부 0.000)이고, 그 기준으로는 **개선 없음**이다.
전환 비율(60%)이 아니라 데이터가 원인이므로 이 잡으로 60% vs 70% 를 논할 수 없다.

---

## 📁 코드 / 데이터셋 상태

### 수정된 코드

| 파일 | 변경 |
|---|---|
| `src/data/convert_hdf5_to_lerobot.py` | **frame fix** (`ROBOT_FRAME_POS`, state/action 동시 이동) + `--use-real-state` |
| `sh/train_pi0_find_hidden.sh` | `USE_REAL_STATE=1` 기본값, 변환기에 플래그 전달 |
| `sh/train_pi0_find_hidden_kisti.sh` | staged-freeze 환경변수 전달 |
| `sh/train_pi0_seld_uv_select_radio.sh` | staged-freeze 환경변수 전달 |
| `third_party/openpi/scripts/train_pytorch.py` | staged-freeze 호출 2줄 |
| `src/staged_freeze/` | **신규** — 학습 중 PaliGemma 동결 전환 (테스트 19개) |
| `src/audio_visual_bridge/` | **신규** — 오디오 (u,v) ↔ 이미지 패치 명시적 브리지 (테스트 32개, 미검증) |
| `sh/run_pi0_kisti_stagedfreeze.slurm`, `sh/run_pi0_kisti_radio_stagedfreeze.slurm` | **신규** |

### 데이터셋 — ⚠️ **frame fix 적용된 것은 하나뿐**

| 경로 | real-state | frame fix | 에피소드 | 비고 |
|---|---|---|---|---|
| `dataset_find_hidden_v3_rxfix_lerobot` | ✅ | ✅ | 880 | **현재 최신본 — v3 + roll 브랜치 수정 (맨 아래 절)** |
| `dataset_find_hidden_v3_lerobot` | ✅ | ✅ | 880 | v3. 감사 PASS지만 roll이 ±π 경계선 위 (맨 아래 절) |
| `dataset_find_hidden_lerobot_framefix` | ✅ | ✅ | 880 | v3 이전의 유일한 정상본 |
| `dataset_find_hidden_lerobot` | ✅ | ❌ | 880 | 프레임 버그 |
| `dataset_find_hidden_top_lerobot` | ✅ | ❌ | 440 | 프레임 버그 (top-only 실험용) |
| `dataset_find_hidden_lerobot_v2_stateeqaction` | ❌ | ❌ | 880 | 원본 v2 |
| `dataset_find_hidden_lerobot_v1` | ❌ | ❌ | 880 | 구 카메라 |
| `dataset_seld_uv_select_radio_lerobot` | ❌ | 해당없음 | 438 | **radio는 프레임 문제 없음** |
| 뉴론 `dataset_find_hidden_lerobot_v2_realstate` | ✅ | ❌ | 880 | 프레임 버그 (865268이 사용) |
| 뉴론 `dataset_seld_uv_select_radio_lerobot` | ❌ | 해당없음 | 438 | 정상 |

원본 HDF5 `dataset_find_hidden_v2_src/` (880ep)는 그대로다 — 재생성 없이 재변환만 하면 된다.

### 체크포인트

| 경로 | 학습 데이터 | 비고 |
|---|---|---|
| `outputs/pi0_find_hidden_framefix/.../20000` | ✅ framefix | **success 0.050** |
| `outputs/pi0_find_hidden_top_only/.../10000` | ❌ 프레임 버그 | 0/100 |
| `outputs/pi0_find_hidden_v2/.../20000` | ❌ state==action | 0/100 |
| `outputs/pi0_find_hidden_lora_kisti{,_4gpu}/.../14000` | ❌ 구 카메라 | 0/100 |
| `outputs/pi0_find_hidden_kisti_stagedfreeze/.../14000` | ❌ 프레임 버그 | 0/100 (60% 전환) |
| `outputs/pi0_seld_uv_select_radio_kisti_stagedfreeze/.../20000` | 정상 | **0.810** (70% 전환, 기준 0.690) |
| `outputs/pi0_find_hidden_bottom_only/.../10000` | ✅ framefix, bottom 440ep | 0/100 (bottom 100ep) |
| `outputs/pi05_find_hidden_v3_stagedfreeze/.../11000` | ✅ **v3** | 학습 중 (pi0.5, 70% 전환) |

### 평가 결과 위치

`sh/train_pi0_find_hidden.sh` 의 `EVAL_DIR`은 여전히 `outputs/eval_find_hidden` 으로 **하드코딩**되어
매번 덮어쓴다. `sh/eval_pi05_find_hidden.sh` 를 쓰면 `eval_<exp>_<step>[_<slots>]` 로 자동 분리된다.

```
outputs/eval_find_hidden_framefix/                    framefix 0.050 (영상 105편)
outputs/eval_pi0_find_hidden_framefix_5000/           framefix step5000 0.000  ← 학습곡선용
outputs/eval_pi0_find_hidden_framefix_20000_left_bottom+right_bottom/   0.030 (bottom 100ep)
outputs/eval_pi0_find_hidden_bottom_only_10000_left_bottom+right_bottom/ 0.000 (bottom 전용 학습)
outputs/eval_find_hidden_kisti_stagedfreeze/          KISTI 60% 전환 0.000 (버그 데이터)
outputs/eval_radio_kisti_stagedfreeze/                KISTI radio 70% 전환 **0.810**
outputs/eval_seld_uv_select_radio_only/               radio baseline 0.690
outputs/eval_find_hidden_v2_step20000/                v2 0.000
outputs/eval_find_hidden_kisti_4gpu/                  KISTI 4장 0.000
outputs/eval_find_hidden_top_only_{left,right}_top/   top-only 0.000 (프레임 버그로 무효)
outputs/eval_top_only_horizon50_left_top/             개루프 진단 (프레임 버그로 무효 — 재측정 필요)
```

⚠️ **v3 이전 체크포인트를 평가할 때는 마이크 카메라가 2**여야 한다 (v3에서 1로 이동).
`sh/eval_pi05_find_hidden.sh` 는 체크포인트의 `assets/<repo_id>/` 를 읽어 자동 판별하고,
모르는 repo_id 면 추측하지 않고 종료한다. 수동 지정은 `MIC_CAM=2`.

---

## 📌 환경 메모

- **로컬**: conda `vlabench` + `third_party/openpi/.venv` (평가에 둘 다 사용). `pytest`는 openpi venv에만 있음.
- **KISTI**: 계정 `x3445a03`, scratch `/scratch/x3445a03`, 파티션 `amd_a100nv_8` (동시 2잡).
  `ssh neuron`(비밀번호+OTP). 시스템 `python3`에는 torch가 없다 — preflight는 반드시 openpi venv로.
- **네트워크**: 대용량 전송은 `--bwlimit` 필수 (7GB를 380Mbps로 내리다 트래픽 경보 발생 이력).
  전용 전송 노드 `neuron-dm.ksc.re.kr` (150.183.150.102/103)이 있으나 별도 OTP 로그인 필요.

---

## 🆕 2026-07-29: pi0.5 · 4슬롯 · KISTI 2×A100 런

위 3번(“framefix 데이터로 KISTI 재학습”)을 실행하면서 두 가지를 바꿨다.
상세는 **`docs/KISTI_pi05_find_hidden.md`**.

| 변경 | 내용 |
|---|---|
| 데이터 | `dataset_find_hidden_lerobot_framefix` (880ep / 85,893프레임, **4슬롯 전부**). 첫 프레임 state `[0.0005 0.2290 0.4301]` 이 평가 실측 `[0.0004 0.2297 0.4319]` 과 일치 확인 |
| staged freeze | 60% → **70%** (`FREEZE_LLM_AT_FRAC=0.7`) |
| backbone | pi0 → **pi0.5**. PyTorch base 를 HF `lerobot/pi05_base` 에서 확보(키 812개가 `PI0Pytorch(pi05=True)` 와 1:1, shape mismatch 0). pi0 폴백 불필요 |
| 평가 | `VLABENCH_HIDDEN_SLOT_LABEL` 이 콤마 목록을 받도록 확장(에피소드마다 균등 추출). `sh/eval_pi05_find_hidden.sh` — 기본 4슬롯, `SLOTS=` 로 부분집합, EVAL_DIR 자동 분리 |

한 번 top-only 로 좁혔다가 **되돌렸다**: 목적이 모델의 **고도각(위/아래 서랍) 능력 측정**이라
그 축을 학습에서 빼면 잴 수가 없다. 평가의 슬롯별 분해가 곧 고도각 성적표다.
top-only 본(`dataset_find_hidden_top_lerobot_framefix`, 440ep, `local/avla_find_hidden_top`)은
대조군용으로 남겨 뒀다.

⚠️ 함정 하나: 기존 VLABench pi05 config 들은 `discrete_state_input=False` 인데,
PyTorch pi0.5 는 `pi05=True` 면 `state_proj` 를 만들지 않으므로 **state 가 모델에 아예 안 들어간다.**
새 config 는 기본값(True)을 그대로 둔다.

**pi0 대조군을 동일 조건으로 같이 돌린다** (batch 64 · lr 2e-4 · 10,000 step · freeze 70% ·
2×A100 · 같은 데이터, backbone 만 다름). 한 스크립트의 `BACKBONE=pi0|pi05` 스위치라 두 arm 이
드리프트할 수 없다.

신규 파일: `sh/run_kisti_findhidden.slurm`(BACKBONE 스위치), `sh/eval_pi05_find_hidden.sh`,
`docs/KISTI_pi05_find_hidden.md`, openpi config `pi05_ft_vlabench_find_hidden_lora` +
`pi0_ft_vlabench_find_hidden_framefix_lora`
(둘 다 repo_id `local/avla_find_hidden_framefix` — 구 버그본들이 쓰는 `local/avla_find_hidden` 과 분리).

## 다음 할 일

1. KISTI staged-freeze 평가 2건 결과 확인 (진행 중)
2. ~~고도각 병목~~ → **stall 병목**. 오디오가 아니다(위 “철회됨” 절).
   teacher-forced 진단으로 관측 불일치 vs compounding error 를 가른 뒤 처방을 정한다.
   슬롯별 성공률 비교는 n≥100 으로 볼 것 — n=41 로는 3% 와 0% 를 구분 못 한다
3. ~~framefix 데이터로 KISTI 재학습~~ → pi0.5 4슬롯 런으로 진행 중 (위 섹션)
4. `EVAL_DIR` 하드코딩 제거 — pi0.5 경로는 `sh/eval_pi05_find_hidden.sh` 로 해결.
   `sh/train_pi0_find_hidden.sh:207` 은 아직 그대로다

---

## 🆕 2026-07-30: v3 데이터셋 재생성 (에피소드 품질 게이트 + 마이크 이전)

`sh/rebuild_find_hidden_v3.sh` → `dataset_find_hidden_v3_src` / `dataset_find_hidden_v3_lerobot`
(repo_id `local/avla_find_hidden_v3`, openpi config `pi0{,5}_ft_vlabench_find_hidden_v3_lora`).

### 🐞 v2 원본에서 발견된 데이터 결함

**근본 원인 하나에서 세 증상이 나온다.** `_soften_cabinet_drawers`가 서랍 slide damping을
50 → 1로 낮춘다(그래야 oracle이 서랍을 당겨 열 수 있다). damping은 **속도**만 억제하므로
서랍을 정적으로 붙잡아 두는 게 아무것도 없다. 리셋 settle 중 숨긴 물체가 떨어지며 **자기 서랍을
스스로 밀어낸다**. 실측: 리셋의 **20~25%** 가 임계를 넘는다 (최악 `open_fraction` 0.118,
성공 임계는 0.13).

| 증상 | 실측 |
|---|---|
| **2초짜리 무의미 에피소드** | 880개 중 **10개** (20~39프레임, EE 이동 ≤0.176m). 정상은 ≥64프레임·≥0.208m — 완전히 분리됨. 20프레임짜리는 대기 prefix만 있고 expert 동작이 0프레임 |
| **서랍이 처음부터 열려 있음** | 위와 같은 원인. 대상 서랍에 걸리면 `drawer_open`이 expert가 움직이기 전에 이미 충족 → 첫 `env.step`이 terminal timestep 반환 → `SkillLib.pick`이 `task_success=True`로 즉시 반환 → 생성기가 "성공"으로 저장. 비대상 서랍이 열려 있으면 **정답이 시각적으로 노출**되는 문제도 있었음 |
| **물체가 밖으로 나옴** | 880개 중 **33개** 가 라벨과 다른 서랍 높이에 있다 (31개는 중간 서랍, 2개는 아래). 최악은 z=0.608 — 테이블 아래로 떨어짐. 오디오 고도각 라벨이 엉뚱한 서랍을 가리킨다 |

**추가로 발견한 4번째 결함 (train/eval 불일치)**: oracle 오디오 GT를 **에피소드 끝**에서
스냅샷하고 있었다. 그 시점엔 서랍이 4~10cm 열려 물체가 함께 나와 있다. 반면 eval은
`_oracle_snapshot`을 **매 policy step 마다** 라이브 env에서 다시 계산한다 — 즉 숨어 있는
위치에서 시작한다. 정책이 어느 서랍으로 갈지 정해야 하는 바로 그 프레임들에서 어긋나 있었다.
(프레임 버그와 같은 종류의 오류다.)

### ✅ 수정

| 항목 | 내용 |
|---|---|
| 리셋 복구 | `close_all_drawers()` — 모든 서랍 qpos/qvel을 0으로 놓고 재settle, 최대 3라운드 반복. 테스트 리셋 32회에서 **관측된 모든 drift를 치유** (최악 0.118 → 0.004). 물체가 아직 움직이는 중이라 1라운드로 안 되는 경우가 있어서 반복이 필요하다 |
| 리셋 게이트 | `validate_hidden_scene()` — 서랍이 여전히 열림 / 성공조건 이미 충족 / 물체가 라벨한 서랍에 안 숨어 있음 → 에피소드 폐기 |
| 롤아웃 후 게이트 | `validate_hidden_episode()` — 프레임 수 <45 또는 EE 이동 <0.20m → 폐기. 접근 *중* 서랍이 열리는 경우(리셋 게이트가 못 잡는다)를 잡는다 |
| oracle GT 시점 | **첫 기록 프레임**으로 이동 (`_build_oracle_sources_meta` + `_oracle_snapshot_dict`) |
| 마이크 | camera 2 → **camera 1** (아래 절) |
| HDF5 크기 | `--slim-hdf5`: 아무도 안 읽는 스트림 제거 → **163MB → 41MB/에피소드** (880개 36GB, 이전 143GB) |

게이트 효과 검증 (가장 어려운 left_bottom 슬롯, 10회 시도): **5개 저장 / 게이트 폐기 0건** /
oracle 실패 5건. 즉 반복 close가 게이트 폐기를 사실상 0으로 만든다.

### 🎤 마이크: camera 2 → camera 1 (저위치·중앙으로 재배치)

고도각 단서는 마이크와 서랍의 **수직 오프셋**이다. camera 2는 z=1.75로 두 서랍보다 훨씬 위에
있어 두 고도각이 뭉친다. camera 1을 `camera_config.json`에서 **(0, −1.05, 1.20), 아래로 10.7°,
fovy 50** 으로 재배치해 두 서랍이 마이크를 위아래로 감싸게 했다.

v2 880개 에피소드에서, SlotEncoder가 실제로 먹는 투영 (u,v) 기준:

| 마이크 pose | u d′ (좌/우) | v d′ (위/아래) | 화면밖 |
|---|---|---|---|
| cam2 (0,−1.05,1.75) — 기존 | 12.35 | 5.54 | 0/880 |
| cam1 **원래 pose** (−0.775,−0.856,1.209) | 9.89 | 4.87 | **12/880** |
| **cam1 재배치 (0,−1.05,1.20)** | **12.96** | **7.72** | 0/880 |

⚠️ **cam1을 원래 pose 그대로 쓰면 안 된다.** x=−0.775로 왼쪽에 치우쳐 있어 왼쪽 캐비닛이 FOV
가장자리에 몰리고, 12개 에피소드는 투영이 화면을 완전히 벗어나 오디오 슬롯이 `(-1,-1)` 센티넬로
마스킹된다 — 그 에피소드는 오디오가 아예 없어진다.

**정책 이미지는 안 바뀐다**: converter의 `DEFAULT_CAM_MAP`은 `{image:2, second_image:0,
wrist_image:3}` 이라 camera 1 이미지는 아무도 안 쓴다. 따라서 framefix 세트 대비 **오디오 라벨과
에피소드 품질만** 다르다 — 깨끗한 대조 실험이 된다.

마이크 카메라는 `src/audio/oracle_sled.resolve_mic_cam_id(task_name)` 한 곳에서만 결정한다
(`TASK_MIC_CAM`, env `VLABENCH_MIC_CAM`로 override). 생성·eval 양쪽이 이 함수를 쓴다.

⚠️ **v3 이전 체크포인트를 평가할 때는 `VLABENCH_MIC_CAM=2` 를 반드시 줘야 한다.** 그 체크포인트의
오디오 라벨은 camera 2에서 만들어졌다.

### 🐞 게이트가 만든 회귀 — 잡아서 고쳤음 (반드시 기억할 것)

`close_all_drawers`가 서랍을 안정시키려고 `env.step()`을 10~30회 추가로 돌린다. 그런데
`LM4ManipDMEnv.step(None)`은 **현재 qpos를 그대로 목표로 명령**한다 — 즉 스텝 사이의 중력
처짐이 새 목표로 승격되는 **래칫**이다. 스텝을 더 돌릴수록 팔이 계속 내려간다. eval은 그런
추가 스텝을 돌리지 않는다.

실측 (6회 리셋, 완전히 결정론적):
```
리셋 직후    base-frame EE  [0.0004, 0.2297, 0.4319]   ← eval이 기록하는 값과 정확히 일치
close 이후                  [0.0006, 0.2224, 0.4137]   z −18.2mm
```
원본 HDF5로도 확인:
```
v2 (close 없음)   첫 프레임 EE [0.0005, 0.2295, 0.4315]   eval과 오차 <0.5mm  ✅
v3 (close 있음)                [0.0007, 0.2214, 0.4115]   y −8.1mm, z −20.1mm  ❌
```
**프레임 버그와 같은 종류의 train/eval 불일치**를 내가 새로 만든 것이었다 (크기만 작다).
28.8cm 버그의 z 성분이 7.4cm였던 걸 생각하면 2cm는 무시할 값이 아니다.

**수정**: `_snapshot_robot_pose` / `_restore_robot_pose` — close 진입 시 모든 로봇 joint의
qpos를 저장하고, 끝나기 전에 되돌린 뒤 `physics.forward()`. 정지 상태의 팔은 캐비닛에서 멀어
접촉이 없으므로 텔레포트가 안전하다. 검증: close 전후 shift가 정확히 `[0, 0, 0]`.

**추가 하드닝**: settle 루프 중 서랍이 성공 임계를 넘으면 그 스텝이 terminal timestep을
반환하고, dm_control `composer.Environment`는 **다음** step에서 조용히 `reset()`을 호출해
씬 전체를 재랜덤화한다. 그러면 기록된 이미지와 저장된 `episode_config`가 어긋난다.
이제 매 스텝 조건을 확인해 즉시 빠져나오고 에피소드를 폐기한다.

⚠️ **교훈: 기록 시작 전에 `env.step()`을 추가로 돌리면 팔이 처진다.** find_hidden 파이프라인에
스텝을 추가하는 변경을 할 때마다 첫 프레임 state를 eval 실측값과 반드시 비교할 것.
`scripts/audit_find_hidden_dataset.py`가 이걸 hard-fail로 검사한다.

**로봇 rest pose는 원래 bimodal이다** (v2 실측 200개: 83%가 base-frame z=0.4315, 17%가
0.4687). `env.reset()`의 settle 특성이며 eval도 같은 분포를 샘플링하므로 문제가 아니다.
다만 LeRobot v2.1은 **에피소드당 parquet 1개**를 쓰므로 감사에서 파일 하나만 읽으면
동전던지기가 된다 — 감사는 여러 에피소드의 **median**을 쓴다.

### 🔒 로봇 base frame — 그대로 유효

씬을 건드리지 않았다 (로봇 base `(0,−0.7,0.7)` 동일, 캐비닛 동일). 따라서
`convert_hdf5_to_lerobot.py`의 `ROBOT_FRAME_POS` 수정이 그대로 적용된다. v3 스크립트는
`USE_REAL_STATE=1`을 넘기고, `scripts/audit_find_hidden_dataset.py`가 변환 후
**첫 프레임 state가 eval 실측 rest pose와 1cm 이내로 일치하는지 hard-fail로 검사**한다.

### 📋 감사 스크립트

`scripts/audit_find_hidden_dataset.py --src-dir <...>/find_hidden_object_open --lerobot-dir <...>`
— v2를 망친 결함들을 그대로 검사한다: 퇴화 에피소드, 물체 배치/서랍 레벨, (u,v) d′와 화면밖 개수,
슬롯 밸런스, LeRobot base frame. 하나라도 걸리면 exit 1.

### 📊 최종 결과 (2026-07-30 완료)

**880 에피소드 (슬롯당 정확히 220개), 87,087 프레임. 감사 PASS.**

| 항목 | 값 |
|---|---|
| HDF5 원본 | `dataset_find_hidden_v3_src` **33GB** (v2는 136GB — 4.1배 감소) |
| LeRobot | `dataset_find_hidden_v3_lerobot` 516MB, repo_id `local/avla_find_hidden_v3` (심링크 생성됨) |
| 프레임 수 | min 68 / p1 85 / med 97 / max 135 (게이트 45) |
| EE 이동 | min 0.392 / med 0.501 / max 0.587 m (게이트 0.20) |
| **첫 프레임 state** | `[0.0005, 0.229, 0.4301]` vs eval 실측 `[0.0004, 0.2297, 0.4319]` → **y 0.7mm, z 1.8mm** ✅ |
| **방위각 d′ (u)** | **16.37** (v2: 12.35) |
| **고도각 d′ (v)** | **16.90** (v2: 5.54 → **3.1배**) |
| 화면밖 | **0/880** |

고도각 개선의 대부분은 마이크 이전이 아니라 **배치 게이트**에서 나왔다. v2는 top 슬롯의
약 22%가 라벨과 다른 위치에 물체가 있었고 그게 고도각 라벨 잡음의 주원인이었다.
슬롯 내 분산이 무너졌다: `el` std 0.57~0.87°, `v` std 0.011~0.017 (분리폭 0.24).

**생성 통계** (총 1198회 시도 → 880개 채택):

| 슬롯 | 시도 | oracle 실패 | 게이트(리셋) | 게이트(롤아웃) | 크래시 |
|---|---|---|---|---|---|
| left_top | 225 | 5 | 95 | 0 | 3 |
| right_top | 226 | 6 | 107 | 0 | 0 |
| left_bottom | 301 | 81 | 36 | 0 | 1 |
| right_bottom | 446 | 225 | 68 | **1** | 3 |

- **리셋 게이트 306건** — v2였다면 서랍이 열린 채로 / 물체가 엉뚱한 곳에 있는 채로 그대로
  데이터셋에 들어갔을 에피소드들. 비율이 런 내내 일정했다 (top 30~32%, bottom 11~13%)
  — v2 오프라인 추정치(21.8% / 6.8%)와 일치.
- **롤아웃 게이트 1건** — 리셋 게이트가 구조적으로 못 잡는 케이스(접근 *중* 서랍이 열림).
  드물지만 제 역할을 했다.
- 크래시 7건은 전부 `mjWARN_BADQACC` (리셋 중, 기존 이슈). 재시도 루프가 흡수.
  어느 슬롯도 시도 예산 근처에도 못 갔다.

### ⚠️ 알려진 잔여 항목 (무해하다고 판단, 근거 포함)

**top 슬롯 물체 구성 편향**: `boxed_food` 19% (기대 33%). 배치 게이트가 물체 높이로
필터링하기 때문이다 — `boxed_food`가 top 서랍에서 가장 불안정해서 중간 서랍 높이
(dz≈0.234)로 떨어지는 경우가 많고, 그건 **정당한 폐기**다.

정책에 영향이 없는 근거 (880개 실측):

| 경로 | 측정 |
|---|---|
| 고도각 | 물체별 `el` 차이 최대 0.79° — 그런데 top/bottom 분리폭은 **12.4°**. 구성 편향이 top 평균을 움직이는 양은 **0.08°** |
| energy | 물체 간 사실상 동일 (0.7212 / 0.7240 / 0.7213) |
| 음원 클래스 | 물체와 독립 — 모든 물체가 36개 중 33~36개 클래스에 걸침 |
| 시각 | 접근 내내 물체는 서랍 안에 숨어 있고, 마지막 10~20프레임(서랍 선택 **이후**)에만 노출 |

감사는 이걸 **실패가 아니라 WARN**으로 보고한다. ⚠️ 단, **물체를 잡아야 하는 태스크**
(composite `find_hidden_object` retrieve)에서는 다시 따져야 한다.

**eval에도 같은 서랍 드리프트가 있다** (수정하지 않았음 — 과거 수치와의 비교 가능성 유지).
드리프트는 ~0.12까지 가지만 성공 임계 0.13을 넘는 경우는 드물어서, **eval 에피소드의
약 1~2%가 정책이 아무것도 안 해도 즉시 성공**으로 집계될 것으로 추정된다 (v2의 880개 중
10개 쓰레기 에피소드와 일치). 측정치 3~5% 대비 상대적으로 20~30% 부풀림이라 슬롯별
비교에서는 유의미하다. eval에도 강제 닫기를 넣으려면 플래그로 넣는 것을 권장.

### 신규/수정 파일

| 파일 | 변경 |
|---|---|
| `VLABench/tasks/.../find_hidden_object_open_series.py` | `close_all_drawers`, `drawer_open_fractions`, `validate_hidden_scene`, `validate_hidden_episode` + 임계 상수 |
| `VLABench/configs/camera_config.json` | find_hidden 두 태스크에 camera `"1"` (마이크) 추가 |
| `src/audio/oracle_sled.py` | `TASK_MIC_CAM`, `resolve_mic_cam_id()` |
| `scripts/trajectory_generation.py` | 게이트 호출, oracle GT 시점 이동, `_build_oracle_sources_meta`, `--slim-hdf5`, `--no-scene-gates`, 마이크 cam 해석 |
| `src/eval/eval_smolvla_audio.py` | `mic_cam_id` (정책 이미지 `FRONT_CAM`과 분리) — `eval_pi05_audio.py`도 여기서 import하므로 함께 적용됨 |
| `VLABench/utils/data_utils.py` | `save_single_data(drop_keys=, split_rgb_per_camera=)`, `SLIM_DROP_KEYS` |
| `sh/gen_find_hidden_balance.sh` | 시도 예산 증가(게이트 폐기 흡수), `SLIM_HDF5` |
| `sh/rebuild_find_hidden_v3.sh` | **신규** — 생성 → 변환 → 감사 |
| `scripts/audit_find_hidden_dataset.py` | **신규** |
| openpi `config.py` | **신규** `pi0_ft_vlabench_find_hidden_v3_lora`, `pi05_ft_vlabench_find_hidden_v3_lora` |

### 📌 정리 메모

`dataset_find_hidden_v2_src` (136GB)는 v3가 감사를 통과하면 삭제 가능하다. 디스크가 90% 차
있었기 때문에 `--slim-hdf5`가 사실상 필수였다 (full-fat 두 번째 사본이면 96%가 된다).

---

## 🆕 2026-07-31: v3는 감사를 통과하는데도 성능이 낮았던 이유 — roll(rx)이 오일러 ±π 경계선 위

계기: **`dataset_seld_uv_mixed`로 학습한 모델은 잘 되는데 `dataset_find_hidden_v3_src`는
같은 조건에서 성능이 매우 낮다** — 태스크가 어려운 건지 데이터 생성이 잘못된 건지 구분이 필요했다.

### ✅ 먼저, v3에서 정상인 것들 (전부 재확인함)

`scripts/audit_find_hidden_dataset.py` → **PASS** (880 ep, 슬롯당 정확히 220개).

| 항목 | 결과 |
|---|---|
| 퇴화 에피소드 | 없음 (최소 68프레임 / EE 이동 0.392m) |
| 물체 배치 vs 슬롯 라벨 | 880/880 일치 |
| 오디오 분리도 | 방위각 d′ **16.37**, 고도각 d′ **16.90**, off-screen 0/880 |
| base frame | 첫 프레임 state가 eval 측정치와 y 0.7mm / z 1.8mm |
| **이미지↔state 시간 정렬** | 손목 카메라 프레임차 vs EE 이동량 상관이 **lag 0에서 최대**(+0.518), 비디오/parquet 프레임 수 정확히 일치 → 프레임 오프셋 없음 |

즉 v3에서 고친 항목(자가개방 드로어 게이트, 마이크 이전, oracle GT 시점, frame fix)은 전부 제대로 들어가 있다.
**감사 스크립트가 검사하지 않던 축에 별개의 결함이 하나 더 있었다.**

### 🐞 결함: roll이 브랜치 컷 위에 앉아 있다

`VLABench/utils/utils.quaternion_to_euler` = scipy `as_euler('xyz')` → roll 은 (−π, π] 로 나온다.
그리퍼의 top-down 홈 자세는 roll ≈ **±180°**, 즉 *정확히 그 경계선 위*다.
`skill_lib`는 웨이포인트마다 `quaternion_to_euler`를 다시 호출하므로 브랜치가 매번 새로 결정된다.

880 에피소드 전수 측정:

```
전체 프레임의 45.5% 가 |rx| > 3.0        (+π 쪽 3.0%, −π 쪽 42.6%)
에피소드의 8~18% 가 rx=+3.141 로 시작    (나머지는 −3.132 — 같은 자세인데 6.27 rad 차이)
70/880 에피소드가 중간에 ±2π 점프 포함
```

여기서 끝이 아니라, **openpi가 action을 state 기준 delta로 학습**하기 때문에 증폭된다
(`LeRobotVLABenchDataConfig`, `_transforms.DeltaActions(make_bool_mask(6, -1))`,
openpi `config.py:469` — 타깃 = `action[t:t+50] − state[t]`):

| | find_hidden v3 | seld_uv_mixed (정상 동작) |
|---|---|---|
| delta-rx 타깃이 ±π 초과(=360° 회전 명령)인 청크 | **2670 / 87087 = 3.07%** | **0 / 69339 = 0%** |
| 정규화 후 그 청크들이 차지하는 rx 타깃 에너지 | **70.4%** | — |
| rx action 정규화 std | 0.664(정상) → **1.170**(부풀려짐) | 0.007 |
| 같은 물리 자세의 state 입력 편차 | **5.37 σ** | 없음 |

**3%의 청크가 rx 회귀 타깃 에너지의 70%를 먹고 있었다.** 그리고 rx는 하필 이 태스크의 핵심
DOF다 — 손잡이를 잡으려면 손목을 top-down에서 face-forward로 돌려야 한다.

**radio 데이터셋이 멀쩡했던 이유:** `dataset_seld_uv_mixed_lerobot`은 손목이 계속 top-down이라
rx std 0.007, 100% −π 브랜치, 래핑 청크 0개다. 손목을 돌리는 태스크가 find_hidden뿐이라
**이 태스크만 경계선을 넘는다.** "동일 조건 비교"가 성립하지 않았던 것.

### ✅ 수정 — 변환 단계만, 33GB 재생성 불필요

`VLABench/utils/utils.fold_roll_to_negative_branch()` 신규: roll을 `(-2π, 0]` 단일 브랜치로 접는다.

```python
return np.where(roll > 0.0, roll - 2.0 * np.pi, roll)
```

**stateless**인 것이 핵심이다 — 현재 프레임만 있으면 되므로 시간축 unwrap 없이
변환기와 평가 루프가 **동일하게** 적용할 수 있다. 이미 음수 브랜치에 있는 자세에는 no-op이라
select_radio 계열 데이터셋은 값이 바뀌지 않는다.

적용 위치 (셋 다 필요):

| 파일 | 위치 |
|---|---|
| `src/data/convert_hdf5_to_lerobot.py` | `_compute_state_action` (action roll), `_compute_real_state` (**action roll + state 오일러 양쪽**) |
| `src/eval/eval_smolvla_audio.py` | `_build_policy_batch`의 `quaternion_to_euler` 직후. 여기를 빼먹으면 새 train/eval 불일치가 생긴다. `eval_pi05_audio.py`도 이 함수를 재사용하므로 함께 커버됨 |

출력 측(`_apply_action`)은 `euler_to_quaternion`을 쓰므로 브랜치를 가리지 않는다 → **수정 불필요**.

재변환 후 실측 (880 에피소드 / 87,087 프레임 전수, 2026-07-31 15:08 완료):

| 지표 | before (`v3_lerobot`) | after (`v3_rxfix_lerobot`) |
|---|---|---|
| delta-rx 타깃이 ±π 초과인 청크 | 2670/87087 (**3.07%**) | **0/87087 (0.00%)** |
| 에피소드 내 ±2π rx 점프 | **70/880** | **0/880** |
| action rx 정규화 std | **1.170** | **0.664** |
| t=0 state rx 분포 폭 | **6.2724 rad** (bimodal −3.1319 / +3.1405) | **0.0203 rad** (단일, −3.1427~−3.1224) |

**부작용 없음을 확인함:** 두 데이터셋의 parquet을 프레임 단위로 대조한 결과
`|old − new|` 최대값이 `[x y z rx ry rz grip] = [0, 0, 0, 6.283185, 0, 0, 0]` —
**rx 외 모든 채널이 완전히 동일**하고, rx가 바뀐 곳의 차이는 정확히 2π다.
즉 이 재변환은 오직 roll 브랜치만 건드렸으므로 기존 v3와 직접 비교 가능하다.

감사도 재실행 → **PASS** (880 ep, 슬롯당 220, 오디오 d′ 16.37/16.90, base frame 오차 y 0.7mm / z 1.8mm).

⚠️ **경고 술어 주의.** `fold_roll_to_negative_branch`는 0 근처 roll에 대해 경고하는데, 처음엔
`|roll| < π/2`로 걸었더니 **오탐**이 났다. find_hidden은 손목이 손잡이를 향할 때 roll이
정상적으로 −1.57 rad까지 간다 — 이 값들은 음수라 fold가 **건드리지 않는다**. fold가 실제로
옮기는 것은 양수뿐이고, 그 최소값은 측정 결과 state 2.72 / action 3.14로 전부 +π 근처였다.
술어를 `0 < roll < π/2`로 고쳤다.

### 📦 재변환 결과

| | 값 |
|---|---|
| 출력 | `dataset_find_hidden_v3_rxfix_lerobot` **516MB, 880ep / 87,087 프레임** (기존 `dataset_find_hidden_v3_lerobot`은 비교용 보존) |
| repo_id | `local/avla_find_hidden_v3_rxfix` (`~/.cache/huggingface/lerobot/local/…`로 심링크됨) |
| 명령 | `DO_CONVERT=1 USE_REAL_STATE=1 GEN_ROOT=dataset_find_hidden_v3_src LEROBOT_DIR=… REPO_ID=… bash sh/train_pi0_find_hidden.sh` |
| 코덱 | AV1 (기존 v3와 동일 — `--vcodec`는 이 LeRobot 버전의 `create()`가 받지 않아 무시된다) |

### ⚠️ 아직 검증되지 않은 것

이 수정 하나로 성능이 오른다는 보장은 **없다**. 위 「진짜 병목: 정지(stall)」 절에서 확인된
실패 양상(손잡이 0.11~0.20m 앞에서 멈춤, progress 정확히 0.000)을 rx 오염이 전부 설명하는지는
**재학습 전에는 미검증**이다. 다만 수정 비용이 변환 단계뿐이라 먼저 돌려볼 가치가 있다.
학습하려면 openpi `config.py`에 `repo_id="local/avla_find_hidden_v3_rxfix"`를 가리키는
TrainConfig가 추가로 필요하다 (아직 없음).

### 📎 결함 아님으로 판명된 것 (기록용)

- **첫 20프레임 idle prefix의 그리퍼 명령이 "닫힘"(`action[:,6]=0.0`)인데 실제 그리퍼는 열려 있다.**
  두 데이터셋 **모두** 동일하므로 성능 차이의 원인이 아니다. 다만 radio는 state==action 누출
  덕분에 라벨과 입력이 서로 일치하고, find_hidden은 `--use-real-state`라 그 20프레임 동안
  라벨이 관측과 모순된다.
- **radio(mixed)는 `state == action`이 정확히 일치한다** (누출). find_hidden v3만 실제 state를 쓴다.
  v3 쪽이 더 올바른 설정이지만, 그래서 두 데이터셋은 애초에 "동일 조건"이 아니다.
- 그리퍼가 손잡이를 문 뒤 `get_ee_open_state`가 계속 "열림"을 보고하는 것(post-close 프레임의 10%만
  닫힘) — 손가락이 손잡이에 막혀 완전히 닫히지 않는 물리적 사실이고, eval이 같은 값을 계산하므로
  train/eval 불일치가 아니다.

### 🚀 KISTI 재학습 제출 (2026-07-31)

기존 대기 **867002 / 867003** (2×A100, framefix 데이터) 취소 → rxfix 데이터로 재제출.

| | |
|---|---|
| 잡 | **870110 = pi0.5**, **870111 = pi0** (`sh/run_kisti_findhidden_rxfix.slurm`, BACKBONE 스위치) |
| 리소스 | **1×A100 80G**, cpu 8, 24h |
| 데이터 | `dataset_find_hidden_v3_rxfix_lerobot` (전송 완료, rsync dry-run 무차이) |
| config | `pi0{,5}_ft_vlabench_find_hidden_v3_rxfix_lora` (KISTI에서 로드 검증) |
| 하이퍼 | batch **64**, lr **2e-4**, **6,000 step**, freeze @ **4,200** (70%), save 600 |

**batch 64 근거:** batch 32가 32GB에 들어가므로 64 ≈ 55GB → 80G A100 안전, 96은 초과.
글로벌 배치가 64로 **변하지 않았으므로 lr 2e-4 유지** — 재스케일 근거 없음.

**6,000 step 근거:** 단일 A100 실측 런 **865268** (batch 32, 16코어, **4.73 s/step**) 기준.
batch 64 → GPU 작업 2배 ≈ 9.5 s/step = 15.8h. 코어가 16→8로 반감된 악조건 ≈ 13 s/step = 21.7h.
둘 다 24h 안에 완주하고, `SAVE_INTERVAL=600`이 전환점 4,200을 정확히 나눠 wall-clock kill이
나도 freeze 이후 체크포인트가 남는다. 6,000×64 = 384k 샘플 = 4.4 epoch (87,087프레임 기준).

스크립트에 **roll-fold preflight**를 넣었다 — 학습 데이터에 양수 브랜치 roll이 하나라도 남아
있으면 하드 페일한다. 규약이 섞이면 평가에서 조용히 틀리기 때문이다.

### ⚠️ 큐 교훈: 대기 시간을 만든 것은 GPU 개수가 아니라 **age 우선순위**였다

취소 직전 867002의 시작 예정은 **07-31 17:40**(1.7시간 뒤)이었는데, 방금 제출한 870110은
**08-02 03:52**(36시간 뒤)로 잡혔다. 원인:

```
PriorityWeightAge       = 1680  (PriorityMaxAge 7일)  → 하루 약 240점
PriorityWeightFairShare = 10000 → 이 계정 기여분 271점

867002 : 07-29 13:10 제출, age 약 2.1일 → 504 + 271 ≈ 775점
870110 : 07-31 제출,      age 0        →   0 + 271 =  271점
```

파티션은 8노드 중 3 alloc / 5 mix로 다른 사용자 잡이 차 있고 순서는 우선순위로 정해진다.
즉 **2×A100이라 밀린 게 아니라 큐에서 2일 묵어 앞순위였던 것**이고, 취소로 그 2일치가 날아갔다.
2 GPU로 되돌려도 복구되지 않는다. **다음부터: 대기 중인 잡을 취소하기 전에 `sprio`로 age 기여분을
먼저 확인할 것.** 설정 변경이 목적이라면 `scontrol update`로 in-place 수정이 가능한지부터 본다.

### 신규/수정 파일

| 파일 | 변경 |
|---|---|
| `VLABench/utils/utils.py` | **신규** `fold_roll_to_negative_branch()` (+ `warnings` import) |
| `src/data/convert_hdf5_to_lerobot.py` | `_compute_state_action` / `_compute_real_state`에서 roll 접기 |
| `src/eval/eval_smolvla_audio.py` | `_build_policy_batch`의 state 오일러에 roll 접기 |
| `sh/run_kisti_findhidden_rxfix.slurm` | **신규** — 1×A100 · batch 64 · 6,000 step · roll-fold preflight |
| openpi `config.py` | `pi0{,5}_ft_vlabench_find_hidden_v3_rxfix_lora` (KISTI 사본에도 동기화) |
