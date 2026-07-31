# pi0.5 · find_hidden(4슬롯) · KISTI 2×A100 학습

> 작성 2026-07-29. 선행 문서: `KISTI_pi0_STATUS.md`(프레임 버그 근본원인), `KISTI_run_pi0.md`(KISTI 운영).

이번 런이 이전과 다른 점:

| # | 변경 | 이유 |
|---|---|---|
| 1 | **staged freeze 70%** | 앞 70%는 LoRA를 포함해 VLM까지 전부 학습, 뒤 30%는 PaliGemma 동결 후 action expert만. 이전 런은 60% 전환이었다 |
| 2 | **backbone pi0 → pi0.5** | 요청. adaRMS 시간주입 + 이산 state 토큰 |
| 3 | 데이터는 **frame fix** 적용본 | `ROBOT_FRAME_POS` 보정. 이게 빠지면 0% 로 되돌아간다 |

**데이터는 4슬롯 전부**(`{left,right} × {top,bottom}`, 880ep / 85,893프레임) 유지한다.
한 번 top-only 로 좁혔다가 되돌렸다 — **모델의 고도각(위/아래 서랍) 능력을 측정하는 것이 목적**이고,
top-only 로 학습하면 그 축 자체가 사라져 측정이 불가능해지기 때문이다.
평가의 슬롯별 분해(`left_top`/`right_top`/`left_bottom`/`right_bottom`)가 곧 고도각 성적표다.

> 참고: 고도각은 물리적으로 단서가 약하다 — 분리 10.1° vs 그룹내 잡음 5.7°(방위각은 25.2° vs 4.3°).
> 즉 bottom 성적이 낮게 나오는 것은 예상 범위이고, 그 **정도**를 재는 것이 이 런의 관측 대상이다.
> top-only 데이터셋(`dataset_find_hidden_top_lerobot_framefix`, 440ep)은 만들어 둔 채로 남아 있으니
> 대조군이 필요하면 바로 쓸 수 있다.

---

## 1. pi0.5 체크포인트 — 어디서 가져왔나

openpi 는 pi0.5 를 지원하지만(`Pi0Config(pi05=True)`), 이 레포에 있던 PyTorch base 는 pi0 뿐이었다
(`checkpoints/pi0_base_primitive_torch`). 레포 `config.py` 의 기존 pi05 항목들은 전부
`/inspire/hdd/...` 라는 **원 저자 서버 경로**를 가리켜 우리 환경에선 못 쓴다.

HuggingFace 에서 openpi PyTorch 레이아웃과 **키가 1:1로 맞는** base 를 찾았다:

```
lerobot/pi05_base    model.safetensors  14,467,165,872 B (fp32, 812 tensors)
```

검증 (`PI0Pytorch(Pi0Config(pi05=True, audio_conditioning=True, audio_slot_uv=True))` 의 state_dict 와 대조):

```
model keys 821 / ckpt keys 812
MISSING  9  = audio_class_embedding + audio_slot_encoder(8개) + embed_tokens
             (앞 8개는 신규 오디오 모듈이라 fresh init 이 정상,
              embed_tokens 는 lm_head 와 tied — safetensors 메타데이터에 공유 관계가 기록돼 있고
              load_model 이 모델 쪽 tie 를 보고 채운다)
UNEXPECTED 0
SHAPE MISMATCH 0
```

즉 **pi0 로 폴백할 필요 없이 진짜 pi0.5 로 간다.** 다른 후보(`ninjaoden/pi05_base_fp32` 는 동일 파일,
`s3y/pi05_droid_pytorch` 는 DROID 파인튠본이라 base 아님)는 쓰지 않았다.

배치 위치(다운로드는 인터넷 되는 로그인 노드에서, 60MB/s 제한):

```
/scratch/x3445a03/AVLABench/checkpoints/pi05_base_torch/{model.safetensors,config.json}
```

fp32 그대로 둔다. 학습 정밀도는 bf16(`pytorch_training_precision`)이고
`safetensors.torch.load_model` → `load_state_dict` 가 `copy_` 로 캐스팅하며,
로드는 mmap 이라 상주 메모리 부담이 아니다.

### ⚠️ `discrete_state_input` 함정

기존 VLABench pi05 설정들은 전부 `discrete_state_input=False` 다. **따라 하면 안 된다.**
PyTorch pi0.5 모델은 `pi05=True` 일 때 `state_proj` 를 아예 만들지 않는다
(`pi0_pytorch.PI0Pytorch.embed_suffix`: `if not self.pi05:` 안에서만 state 를 임베딩).
따라서 `discrete_state_input=False` 면 **state 가 모델에 전혀 들어가지 않는다** —
프로프리오셉션 없는 정책이 된다. 새 config 는 기본값(=`pi05`=True)을 그대로 둔다.

새 config: `pi05_ft_vlabench_find_hidden_lora` (`third_party/openpi/.../training/config.py`)
- `repo_id=local/avla_find_hidden_framefix` — pi0 config 의 `local/avla_find_hidden` 과 일부러 분리했다.
  그 id 는 **frame 버그본**들도 함께 쓰고 있고 `HF_LEROBOT_HOME` 심링크는 매 런 덮어써진다.
  이 id 는 오직 frame fix 본만 가리킨다
- `audio_mode="slots_uv"` — 오디오 경로는 prefix 에 토큰을 붙이는 방식이라 backbone 과 무관, 그대로 동작
- weight env 는 `OPENPI_PI05_PYTORCH_WEIGHT` (pi0 것과 섞이지 않게 별도 변수)

---

## 2. 데이터 — 4슬롯 + frame fix

```
dataset_find_hidden_v2_src/find_hidden_object_open/   880ep, 슬롯당 220
  → src/data/convert_hdf5_to_lerobot.py --use-real-state (frame fix 내장)
  → dataset_find_hidden_lerobot_framefix/   880ep / 85,893프레임
     repo_id=local/avla_find_hidden_framefix
```

검증: 첫 프레임 state `[0.0005 0.2290 0.4301]` ↔ 평가 시뮬 실측 `[0.0004 0.2297 0.4319]`,
`state != action`(real-state 반영됨).

같은 변환기로 만든 top-only 본 `dataset_find_hidden_top_lerobot_framefix`(440ep / 40,952프레임,
`local/avla_find_hidden_top`)도 남아 있다 — 대조군용. **`dataset_find_hidden_top_lerobot`(구본)과
`dataset_find_hidden_lerobot`(구본)은 frame 버그가 있으므로 쓰지 않는다.**

## 3. 평가 — 기본은 4슬롯 전부

```bash
POLICY_DIR=outputs/pi05_find_hidden_stagedfreeze/<config>/<exp>/<step> \
  bash sh/eval_pi05_find_hidden.sh              # SLOTS 미지정 = 4슬롯
```

슬롯별 성공률 분해가 그대로 **고도각 성적표**다. 특정 축만 보고 싶으면:

```bash
SLOTS=left_bottom,right_bottom ...   # 고도각 스트레스 테스트
SLOTS=left_top,right_top       ...   # 방위각만
```

`VLABENCH_HIDDEN_SLOT_LABEL` 이 이제 **콤마 목록**을 받아 에피소드마다 균등 추출한다.
이전에 특정 슬롯 조합을 보려면 슬롯당 한 번씩 돌려야 했다
(`outputs/eval_find_hidden_top_only_{left,right}_top`). 단일 라벨 동작(생성기 경로)은 그대로다.
`sh/eval_pi05_find_hidden.sh` 는 `EVAL_DIR` 을 체크포인트·슬롯별로 자동 분리해
`sh/train_pi0_find_hidden.sh:207` 의 덮어쓰기 문제도 피한다.

---

## 4. 하이퍼파라미터 근거

기준점: 로컬 framefix 런 = **같은 데이터**(85,893 프레임 4슬롯)에서 batch 16 × 20k step
= 3.7 epoch, lr 1e-4, success 0.050.

| 항목 | 값 | 근거 |
|---|---|---|
| GPU | 2 × A100 80G (DDP) | 요청. `amd_a100nv_8`, GPU당 8코어 → `--cpus-per-task=16` |
| 글로벌 배치 | **64** (32/GPU) | 로컬 16의 4배. 32/GPU 는 job 865268(1×A100, pi0, batch 32)에서 실측 검증됨 |
| lr | **2e-4** | batch 4배 → sqrt 스케일. warmup 1000, cosine decay → 1e-5 |
| steps | **10,000** (= 640k 샘플 = 7.5 epoch) | 로컬 3.7 epoch 의 2배. 아래 참조 |
| staged freeze | `FREEZE_LLM_AT_FRAC=0.7` | 10,000 → step 7,000 전환 |
| save interval | 1,000 | 7,000(전환 지점)에 정확히 걸리고 체크포인트 10개로 best 선택. 1개 ≈ 7GB |

### 10,000 step 이면 충분한가

**충분하다고 본다. 근거 세 가지.**

1. **같은 데이터에서 이미 수렴을 봤다.** batch16×20k(=3.7 epoch)에서 loss 0.4 → 0.008.
   10,000×64 = 640k 샘플 = **7.5 epoch** 으로 그 두 배를 본다.
2. **step 은 이미 배제된 레버다.** `KISTI_pi0_STATUS.md` 의 가설 배제표에서
   “데이터 부족 / 학습 부족” 항목은 880ep·20k, KISTI 4장 20epoch 로 반증됐다 —
   전부 0% 였고 원인은 프레임 버그였다. 지금 성능을 가르는 건 step 수가 아니라
   (a) frame fix, (b) 고도각 정보량이다.
3. **비교 실험의 제약.** pi0 arm 과 step 수가 같아야 한다. 한쪽만 늘리면 backbone 비교가 아니라
   예산 비교가 된다.

**위로 못 늘리는 이유는 24h 벽이다.** 실측 s/step 별 상한:

| s/step | 10,000 소요 | 24h 안에 가능한 최대 step |
|---|---|---|
| 5.0 | 13.9h | 17,280 |
| 6.0 | 16.7h | 14,400 |
| 7.0 | 19.4h | 12,340 |
| 8.5 | 23.6h | 10,160 |

pi0.5 는 pi0 보다 느리다 — `max_token_len` 48 → 200 이라 prefix 가
768(이미지)+48 → 768+200 으로 늘어난다(≈ +18% 시퀀스).
실측 기준점은 pi0 · 1×A100 · batch 32 = 4.87 s/step (job 865268).
10,000 은 **최악의 경우(8.5 s/step)에도 완주**하는 값이다.

> 스모크(866929)가 실제 s/step 을 재고 나면, 여유가 있을 경우 두 arm 을 **같이**
> 올릴 수 있다 (예 6 s/step 이면 14,000 까지). 본 런은 하루 뒤에나 시작하므로
> 그 전에 `scancel` → `sbatch --export=ALL,TRAIN_STEPS=14000,...` 로 교체할 시간이 있다.
> 단 `SAVE_INTERVAL` 도 `TRAIN_STEPS×0.7` 의 약수로 맞춰야 전환 지점에 체크포인트가 걸린다
> (14,000 → 전환 9,800 → `SAVE_INTERVAL=1400`).

---

## 4-B. pi0 대조군 (동일 조건)

같은 스크립트의 `BACKBONE=pi0` 경로. **backbone 말고는 전부 동일**하다 —
데이터·batch 64·lr 2e-4·10,000 step·freeze 70%·2×A100·seed.

```bash
ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch --export=ALL,BACKBONE=pi0 sh/run_kisti_findhidden.slurm'
```

config `pi0_ft_vlabench_find_hidden_framefix_lora` 는 기존 `pi0_ft_vlabench_find_hidden_lora` 와
모델은 같고 `repo_id` 만 `local/avla_find_hidden_framefix` 로 바꾼 것이다
(기존 id 는 frame 버그본들과 공유돼 어떤 데이터로 학습했는지 사후 확인이 불가능하다).

비교 대상: 이 pi0 arm 은 로컬 framefix 런(batch16×20k, 1GPU, success **0.050**)의
배치·backbone만 다른 버전이므로, pi0.5 와의 차이뿐 아니라 **배치/스텝 스케일업 자체의 효과**도
같이 읽힌다.

---

## 5. 실행

```bash
# 스모크(2GPU, 30 step, 15에서 DDP 동결 전환 검증 + s/step 측정)
#   --time=00:40:00 이 중요: a100nv_8 가 꽉 차 있을 때 24h 짜리는 하루 넘게 대기하지만
#   짧은 잡은 backfill 로 훨씬 먼저 들어간다.
ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch --time=00:40:00 --export=ALL,SMOKE=1 sh/run_kisti_findhidden.slurm'

# 본 런 (두 arm)
ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/run_kisti_findhidden.slurm'                           # pi0.5
ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch --export=ALL,BACKBONE=pi0 sh/run_kisti_findhidden.slurm'  # pi0
ssh neuron 'squeue -u x3445a03'
ssh neuron 'tail -f /scratch/x3445a03/AVLABench/logs/fh_<jobid>.out'
```

> a100nv_8 는 동시 **2잡** 실행 / 4잡 제출 제한이다. pi0.5 와 pi0 두 arm 이 딱 2잡이라
> 나란히 돌 수 있다.

로그에서 확인할 것:
```
[load] 9 param(s) not in checkpoint (kept fresh init): [... audio_class_embedding ...]   ← 정상
[staged-freeze] armed: PaliGemma freezes at step 7000/10000
[staged-freeze] step 7000: PaliGemma frozen (2942.9M params, 19.6M newly frozen)          ← LoRA 어댑터까지 동결
```

스모크가 검증하는 것: pi0.5 가중치 로드 · LoRA 주입 · **DDP 하에서의 동결 전환** ·
32/GPU OOM 여부 · 실제 s/step. 특히 동결 전환은 본 런에서 14h 쯤 지나 발생하므로
여기서 먼저 깨뜨려 보는 편이 싸다.

학습 후 체크포인트 회수 → 로컬에서 `sh/eval_pi05_find_hidden_top.sh`.
평가는 헤드리스 노드 MuJoCo 렌더링 위험 때문에 KISTI 잡에 넣지 않는다.
