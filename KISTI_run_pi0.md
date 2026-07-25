# KISTI Neuron에서 pi0 find_hidden 학습 (4×A100 / batch 128 / 50k / LoRA)

> `sh/train_pi0_find_hidden_kisti.sh`(신규, 독립) + `sh/run_pi0_kisti.slurm`로 KISTI Neuron에서
> pi0 find_hidden을 **4×A100 DDP**로 돌리는 가이드. 원본 `sh/train_pi0_find_hidden.sh`는 안 건드림.
> 목표: batch 16→**128**, steps 20k→**50k**, PaliGemma **LoRA**.
> 로그인/OTP/데이터 전송 인증은 **직접**.

계정 `x3445a03` · scratch `/scratch/x3445a03` · 로그인노드 `glogin01`
접속: 별도 터미널에서 `ssh neuron` (마스터 커넥션 12h — 최초 1회만 OTP).

---

## 0. 먼저: 홈 quota 해제 (안 하면 sbatch 자체가 거부됨)

현재 `/home01/x3445a03` = **260G / 64G 초과** → 쓰기·잡 제출 전부 막힘. 범인은 `chaerin`(214G).

```bash
ssh neuron 'mv /home01/x3445a03/chaerin /scratch/x3445a03/chaerin'   # 별도 마운트라 실복사, 오래 걸림
ssh neuron 'lfs quota -u x3445a03 /home01'                            # 64G 밑으로 내려갔는지 확인
```

> 이후 **모든 것(코드·데이터·venv·체크포인트·캐시)을 `/scratch`에**. `chaerin` 이동/삭제는 본인 데이터라 직접 판단.
> sbatch가 `HF_HOME`·`UV_CACHE_DIR`·`HF_LEROBOT_HOME`을 전부 scratch로 돌려놓음(기본값이 홈이라 재초과 방지).

---

## 1. 전송 (핵심: 78G HDF5 원본은 안 보냄)

이 KISTI 스크립트는 **변환·생성·평가 단계가 없다**(norm + train만). 이미 변환된 LeRobot 데이터(218M)만 있으면 됨.

| 대상 | 로컬 경로 | 크기 |
|------|-----------|------|
| 변환된 데이터셋 | `dataset_find_hidden_lerobot/` | **218M** |
| JAX base weight | `checkpoints/pi0_base_primitive/params/` | 12G |
| PyTorch base weight | `checkpoints/pi0_base_primitive_torch/` | 6.6G |
| AVLABench + openpi 코드 | 레포(단 `.venv`·`_src`·`outputs` 제외) | 수백 MB |
| ~~HDF5 원본~~ | ~~`dataset_find_hidden_lerobot_src/` (78G)~~ | **안 보냄** |

```bash
cd /home/rllab/Desktop/AVLABench
DEST=neuron:/scratch/x3445a03/AVLABench
rsync -avP --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
      --exclude 'dataset_find_hidden_lerobot_src' --exclude 'outputs' ./ $DEST/
rsync -avP dataset_find_hidden_lerobot/          $DEST/dataset_find_hidden_lerobot/
rsync -avP checkpoints/pi0_base_primitive/       $DEST/checkpoints/pi0_base_primitive/
rsync -avP checkpoints/pi0_base_primitive_torch/ $DEST/checkpoints/pi0_base_primitive_torch/
```

> `third_party/openpi/.venv`는 로컬 CUDA용 → 전송 금지, KISTI에서 새로 빌드(§3).
> 위 rsync에 이 두 신규 파일(`sh/train_pi0_find_hidden_kisti.sh`, `sh/run_pi0_kisti.slurm`)도 함께 올라감.

---

## 2. 신규 스크립트가 원본과 다른 점 (패치 불필요)

원본을 수정하는 대신 독립 스크립트 `sh/train_pi0_find_hidden_kisti.sh`를 새로 만들었다. 차이:

1. **멀티GPU 런처 수정** — 원본은 `torchrun ... python train.py`로 torchrun이 `python`을 스크립트로 오인(NUM_GPUS>1 미검증 흔적). 신규는 torchrun 시 `python`을 뺌.
2. **`PEAK_LR`/`WARMUP_STEPS` 노출** — 원본은 `DECAY_STEPS`만. lr 조정 가능.
3. **gen/convert/eval 제거** — norm stats + train만. (평가는 로컬에서 §7)
4. 캐시 경로는 sbatch에서 scratch로 export.

---

## 3. 환경 빌드 — **로그인 노드에서** (compute 노드는 인터넷 없음)

openpi는 `uv`로 의존성(pytorch-cu128, python≥3.11)을 받는다. 다운로드는 인터넷 있는 로그인 노드에서 끝내야 함.

```bash
ssh neuron          # 로그인 노드(별도 터미널)
export UV_CACHE_DIR=/scratch/x3445a03/uv-cache
# uv 설치(홈 안 쓰게 scratch로)
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/scratch/x3445a03/bin sh
export PATH=/scratch/x3445a03/bin:$PATH
cd /scratch/x3445a03/AVLABench/third_party/openpi
uv sync             # 의존성 다운로드 → .venv 생성 (여기서만 인터넷 사용)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> 로그인 노드에서 **학습 금지**(공용·정책). 여기선 venv 빌드만. `uv sync`는 GPU를 안 잡음.

---

## 4. 제출 (4×A100)

```bash
ssh neuron 'cd /scratch/x3445a03/AVLABench && mkdir -p logs && sbatch sh/run_pi0_kisti.slurm'
ssh neuron 'squeue -u x3445a03'
```

`sh/run_pi0_kisti.slurm`에 이미 박혀 있는 설정:
- `--partition=amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 --time=2-00:00:00 --comment pytorch`
- `NUM_GPUS=4 BATCH_SIZE=128`(→32/GPU) `NUM_WORKERS=8`(4×8=32코어 상한) `TRAIN_STEPS=50000 SAVE_INTERVAL=5000`
- `PEAK_LR=2e-4 WARMUP_STEPS=2000 DECAY_STEPS=50000` · `VLM_LORA=1 FREEZE_PALIGEMMA=0`

**자원 현황(제출 시점 확인)**: `amd_a100nv_8`는 내 한도 동시 2잡·GPU당 8코어. 한 노드에 4장이 비어야 시작되므로, 다 차 있으면 잠시 `PENDING(Resources)`. `ssh neuron 'showque'`로 여유 확인.
- 대안 파티션 `amd_a100_4`(gpu45 전용 4장, GPU당 16코어): 비면 4장 딱 맞음 → slurm에서 `--partition=amd_a100_4 --cpus-per-task=64`, 스크립트 `NUM_WORKERS=16`으로.

---

## 5. ⏱️ TRAIN_STEPS 선정 + 학습 시간

**데이터셋**: 480 에피소드 / **48,183 프레임(=샘플)** / 단일 태스크.
**원본 기준점**: batch16 × 20k = 320k 샘플 = **6.6 에폭**에서 loss 0.4→0.008 수렴.

batch를 128로 키우면 같은 step이라도 **8배 많은 데이터**를 본다:

| steps (batch 128) | 샘플 | 에폭 | 비고 |
|---|---|---|---|
| 10k | 1.28M | 27 ep | 하한 |
| **15k (기본값)** | 1.92M | **40 ep** | **권장** |
| 20k | 2.56M | 53 ep | 여유(과할 수 있음) |
| 50k | 6.4M | 133 ep | ❌ 과적합·낭비 |

> **핵심**: batch가 커져서 20k가 "간신히"가 아니라 오히려 **원본의 8배(53ep)** 라 과함에 가깝다.
> 단일 48k-프레임 태스크엔 **15k(40ep)면 충분**. 성능이 아쉬우면 레버는 step이 아니라 **에피소드 추가**(step↑는 같은 데이터 재탕일 뿐).
> 스크립트 기본값을 **15k**로 설정. 2.5k마다 체크포인트 저장 → 10k·12.5k·15k를 로컬 평가로 비교해 best 선택.

**시간**(로컬 실측 2.4 s/step @batch16·1GPU → 4×A100 per-GPU 32는 ~5 s/step 추정):

| steps | 예상 시간 | 48h 한 잡? |
|---|---|---|
| **15k** | **~21h** | ✅ (resume 불필요) |
| 20k | ~28h | ✅ |
| 30k | ~42h | ✅ (아슬) |
| 50k | ~70h | ❌ 2회 필요 |

기본 15k면 한 번에 끝난다:
```bash
ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/run_pi0_kisti.slurm'
```
더 길게(예 30k) 돌리다 48h를 넘기면 이어서: `sbatch --export=ALL,RESUME=1,DO_NORM=0 sh/run_pi0_kisti.slurm`

---

## 6. 스모크 테스트 먼저 (권장, 멀티GPU 런처 검증)

본 제출 전 interactive(최대 8h)로 몇 스텝만 — 신규 스크립트의 torchrun 런처가 실제로 도는지 확인:

```bash
ssh neuron
salloc --partition=amd_a100nv_8 --gres=gpu:2 --cpus-per-task=16 --comment pytorch --time=00:40:00
# (잡히면)
cd /scratch/x3445a03/AVLABench
export PATH=/scratch/x3445a03/bin:$PATH UV_CACHE_DIR=/scratch/x3445a03/uv-cache
export HF_HOME=/scratch/x3445a03/.cache/huggingface HF_LEROBOT_HOME=$HF_HOME/lerobot HF_HUB_OFFLINE=1
NUM_GPUS=2 BATCH_SIZE=16 NUM_WORKERS=4 TRAIN_STEPS=20 SAVE_INTERVAL=20 \
PEAK_LR=2e-4 WARMUP_STEPS=5 DECAY_STEPS=20 DO_NORM=1 \
OUTPUT_DIR=/scratch/x3445a03/AVLABench/outputs/smoke \
OPENPI_PI0_JAX_WEIGHT=$PWD/checkpoints/pi0_base_primitive/params \
OPENPI_PI0_PYTORCH_WEIGHT=$PWD/checkpoints/pi0_base_primitive_torch \
bash sh/train_pi0_find_hidden_kisti.sh
```

- `torchrun: can't open file 'python'` 류 → 런처 문제(신규 스크립트엔 없어야 정상).
- **CUDA OOM** → 32/GPU가 80G에 안 맞는 것. `BATCH_SIZE=64`(16/GPU)로 낮추거나 GPU 수 늘림. (32/GPU는 로컬 16/GPU의 2배라 여유 있을 것으로 보지만 실측 필요.)

---

## 7. 평가는 로컬에서 (KISTI 잡엔 평가 없음)

`DO_EVAL`은 VLABench MuJoCo 시뮬 → 헤드리스 노드에서 렌더링 실패 위험. 학습 끝나면 체크포인트만 회수:

```bash
rsync -avP neuron:/scratch/x3445a03/AVLABench/outputs/pi0_find_hidden_lora_kisti/ \
      /home/rllab/Desktop/AVLABench/outputs/pi0_find_hidden_lora_kisti/
# 로컬에서 원본 스크립트로 평가만: DO_CONVERT=0 DO_NORM=0 DO_TRAIN=0 DO_EVAL=1 EVAL_STEP=<step> ... bash sh/train_pi0_find_hidden.sh
```

---

## 8. 모니터링

```bash
ssh neuron 'squeue -u x3445a03'
ssh neuron 'showque'                                              # 파티션 여유/내 한도
ssh neuron 'tail -f /scratch/x3445a03/AVLABench/logs/pi0_<jobid>.out'
ssh neuron 'scancel <jobid>'
```

---

## 9. 체크리스트

- [ ] 홈 quota 해제(`chaerin` 214G 이동) — **최우선**
- [ ] 코드 + `dataset_find_hidden_lerobot`(218M) + 두 weight 폴더 scratch로 (`.venv`·`_src` 제외)
- [ ] 로그인 노드에서 `uv sync`로 venv 빌드(인터넷 필요, 여기서만)
- [ ] 스모크 테스트(`NUM_GPUS=2 BATCH_SIZE=16 TRAIN_STEPS=20`)로 런처·OOM 확인
- [ ] `sbatch sh/run_pi0_kisti.slurm` (기본 15k ≈ 21h, 한 잡으로 완주)
- [ ] 학습 후 체크포인트 로컬 회수 → 로컬에서 10k·12.5k·15k 비교 후 best 선택

---

## 부록: 원래 명령 대비 변경

| 항목 | 원래 | 변경 | 이유 |
|------|------|------|------|
| DO_CONVERT | 1 | **제거(0)** | 변환본(218M) 존재 → 78G 원본 전송·변환 불필요 |
| GPU 수 | (미지정=1) | **4** | batch 128은 단일 A100 OOM. 4장=32/GPU. (8장은 자원상 비현실적) |
| lr | 1e-4 | **PEAK_LR=2e-4** | batch 4배 → sqrt 스케일(LoRA 안정성). 글로벌 배치 기준이라 GPU 수와 무관 |
| DO_EVAL | 1 | **제거(0)** | 헤드리스 노드 MuJoCo 렌더링 위험 → 로컬 평가 |
| 실행 방식 | 원본 sh 직접 | **신규 KISTI sh + slurm** | 런처 버그 회피 + scratch 캐시 + 단계 축소 |
| steps | 50k | **15k(~40ep)** | 데이터 48k프레임: 원본 6.6ep 수렴 기준 15k면 충분. 50k=133ep 과함 |
| 소요 | — | **~21h(15k)** | 48h 한 잡으로 완주. 50k였다면 ~70h·2회 |
