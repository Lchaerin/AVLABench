# audio_visual_bridge — 소리 위치 (u, v)와 이미지 패치의 명시적 연결

`src/models`, `src/audio`, `third_party/openpi` 어느 것도 수정하지 않는 독립 패키지입니다.
기존 코드는 이 패키지를 import 하지 않고, 이 패키지도 기존 코드를 import 하지 않습니다.

---

## 1. 왜 필요한가

현재 경로(`audio_mode="slots_uv"`)에서 소리 위치는 이렇게 전달됩니다:

```
audio_token = class_embedding(class_id) + SlotEncoder(u, v, energy, conf)
prefix = [img₀ 패치 256개][img₁ 패치][img₂ 패치][언어 토큰][audio_token …]
```

`SlotEncoder`는 `(u, v)`를 Fourier 특징으로 바꿔 **하나의 토큰**에 넣고, 그 토큰은 이미지
토큰들 **뒤에** 붙습니다. 즉 모델 입장에서 `u = 0.31`이라는 숫자와 이미지 패치
`(row 7, col 5)`가 같은 장소를 가리킨다는 사실을 알려주는 구조가 **전혀 없습니다.**
그 대응은 순전히 모방 학습 데이터에서 암묵적으로 발견해야 합니다.

find_hidden 태스크에서 실제로 측정한 결과, 그 발견은 **강한 축에서만** 일어납니다:

| 단서 | 슬롯 간 분리 | 그룹내 잡음 | 분리/잡음 | 정책이 배웠나 |
|---|---|---|---|---|
| 방위각 (좌/우 캐비닛) | 25.18° | 4.32° | **5.8** | ✅ 종점이 67cm 분리 |
| 고도각 (위/아래 서랍) | 10.10° | 5.71° | **1.8** | ❌ 종점이 3.5cm 분리 (필요: 27cm) |

암묵적 학습은 신호 대 잡음비가 충분할 때만 작동합니다. 이 패키지는 그 대응을
**구조로 고정**해서, 약한 축도 쓸 수 있게 만드는 것이 목적입니다.

---

## 2. 좌표 규약

`src/audio/projection.doa_to_uv`와 반드시 일치해야 합니다:

```
u = x_pixel / W    0 → 이미지 왼쪽,  1 → 오른쪽
v = y_pixel / H    0 → 이미지 위쪽,  1 → 아래쪽    (OpenCV, Y가 아래)
```

PaliGemma 비전 타워는 `N = G×G` 패치를 **row-major**로 내보내므로, 패치 `p = row·G + col`의
정규화 중심은 `((col+0.5)/G, (row+0.5)/G)`입니다. 224×224 / patch 14 → `G = 16`, `N = 256`.
전체 패키지가 이 대응 하나 위에 세워져 있습니다 (`geometry.py`).

⚠️ **카메라를 반드시 맞춰야 합니다.** 투영 기준은 `cam_2`(= `base_0_rgb` = 마이크 = uv 기준)
입니다. 다른 카메라의 패치 토큰을 넘기면 소리를 엉뚱한 화면에 접지시키며, 조용히 잘못됩니다.

---

## 3. 네 가지 브리지

전부 독립이고 조합 가능하며, 전부 `tanh` 게이트가 걸려 있습니다.
**게이트 0이면 수학적으로 완전한 no-op**입니다 (테스트로 보장). 이미 학습된 체크포인트에
붙여도 step 0에서 아무것도 바뀌지 않습니다 — 기존 `SlotEncoder`가 쓰는 Flamingo식 게이팅과
같은 방식입니다.

### B1. `PatchAnchoredSlot` — 오디오 토큰 ← (u,v) 위치의 시각 특징

```
slot_residual = gate · LN(MLP( bilinear_sample(patch_tokens, u, v) ))
```

오디오 토큰이 "추상 좌표에 있는 소리"에서 **"카메라가 저기서 보고 있는 저것에서 나는 소리"**로
바뀝니다. 오디오 단서에서 시각 증거까지 가는 가장 짧은 경로 — 이중선형 보간 한 번입니다.

### B2. `SoundMarkerOnPatches` — 이미지 패치 ← (u,v)에 칠해진 마커

```
w_p = exp(-‖patch_center_p - (u,v)‖² / 2σ²)          (σ 학습 가능)
patch_residual_p = gate · Σ_slots w_p · marker(class_id) · f(energy)
```

소리 위치를 **픽셀과 같은 텐서 안에** 하이라이트로 그려 넣습니다. 모델이 이미지에서 소리
위치를 *찾을* 필요가 없어집니다.

**약한 단서를 겨냥한 브리지가 이것입니다.** 두 숫자로는 분리되지 않는 10°의 고도각 차이도,
16×16 그리드 위에 칠해 놓으면 서로 다른 하이라이트 위치가 됩니다.

### B3. `SpatialAttentionBias` — 오디오 질의 → 패치 키의 어텐션 사전확률

softmax 이전에 더해지는 `[B, K, N]` 가산 바이어스를 만듭니다. 잔차 스트림에는 아무 신호도
넣지 않고, 오디오 토큰이 **처음부터** 자기 위치의 패치를 보게만 합니다.

`b3_max_logit_bias`(기본 4.0)로 상한이 걸려 있어 하드 마스크가 될 수 없습니다 — DOA 추정이
틀렸을 때 회복 가능해야 하기 때문입니다.

### B4. `SharedGridPositionalCode` — 양쪽이 읽는 하나의 2D 코드

`G×G×W` 학습 테이블. 패치 `p`는 `p`행을 그대로 받고, 오디오 슬롯은 **같은 테이블**을 연속
좌표 `(u,v)`에서 이중선형 보간해 받습니다. 두 스트림이 서로 무관한 두 좌표계(패치 순서 vs
Fourier(u,v)) 대신 **하나의 기저**를 공유하게 됩니다.

---

## 4. 어떤 증상에 어떤 브리지

| 증상 | 권장 |
|---|---|
| 방위각은 되는데 고도각이 안 됨 (지금 상황) | **B2 + B4** — 약한 축을 공간에 직접 그림 |
| 소리 위치와 무관한 물체를 조작함 | **B1 + B3** — 오디오를 시각 증거에 묶음 |
| 처음부터 오디오를 아예 안 쓰는 것 같음 | **B3** — 게이트를 키우면 강제 정렬 |
| 최소 위험으로 먼저 시험 | **B2 단독** (8만 파라미터) |

기본값은 **B1 + B2 + B4** (B3는 off — `attach_to_pi0`가 어텐션 마스크에 닿지 못하므로
명시적 편집 경로에서만 쓰세요).

---

## 5. 비용

pi0 폭 2048, 클래스 38, G=16 기준 (실측):

```
b1   2,107,905
b2      79,876
b3           2
b4     524,289
──────────────
합계 2,712,072   (pi0 ~3.3B의 0.082%)
```

LoRA 어댑터(19.6M)의 1/7 수준입니다.

---

## 6. 연결 방법

### 경로 1 — 런타임 부착 (파일 수정 없음, 실험/ablation용)

```python
from src.audio_visual_bridge import BridgeConfig, attach_to_pi0

cfg = BridgeConfig(d_model=2048, n_classes=38, b4_grid=16,
                   b1_gate_init=0.0, b2_gate_init=0.0, b4_gate_init=0.0)  # 체크포인트 이어붙일 때
bridge = attach_to_pi0(model, cfg, image_slot=0)   # image_slot=0 = cam_2
```

pi0의 `embed_prefix` / `embed_image` / `_embed_audio_slots` 세 메서드를 감쌉니다.
`model.audio_visual_bridge`로 등록되므로 체크포인트에 함께 저장됩니다.

**전제**: pi0가 `embed_prefix` 안에서 모든 카메라의 `embed_image`를 `_embed_audio_slots`
**전에** 호출한다는 것 (현재 vendored openpi 리비전에서 참). `attach_to_pi0`가 부착 시점에
검사하고, 순서가 깨지면 조용히 넘어가지 않고 `RuntimeError`를 냅니다.

### 경로 2 — 명시적 편집 (설정이 확정된 뒤 권장)

`third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py`의 `embed_prefix`에서,
이미지 임베딩 루프와 오디오 슬롯 임베딩 사이에:

```python
from src.audio_visual_bridge import unpack_audio_slots

# --- 이미지 루프 직후, embs[0] 이 cam_2 패치 토큰 ---
if audio_slots is not None and getattr(self, "audio_visual_bridge", None) is not None:
    f = unpack_audio_slots(audio_slots)
    av = self.audio_visual_bridge(
        patch_tokens=embs[0], slot_emb=embs[0].new_zeros(
            embs[0].shape[0], f["u"].shape[1], embs[0].shape[-1]),
        u=f["u"], v=f["v"], present=f["present"],
        class_id=f["class_id"], energy=f["energy"])
    embs[0] = embs[0] + av.patch_residual
    self._av_slot_residual = av.slot_residual      # 아래에서 사용

# --- audio_emb, present = self._embed_audio_slots(...) 직후 ---
audio_emb = audio_emb + self._av_slot_residual.to(audio_emb.dtype)
```

B3를 쓸 경우 `scatter_attention_bias(att_bias, av.attention_bias,
patch_start=<첫 패치 인덱스>, slot_start=<첫 오디오 슬롯 인덱스>)`로 어텐션 마스크에
넣으세요. 마스크는 반드시 가산형 float이어야 합니다 (bool이면 `TypeError`).

---

## 7. 학습 시 주의

- **게이트를 로깅하세요.** `bridge.gates()`가 초기값에 붙어 있으면 브리지가 실제로는
  사용되지 않는 것이고, 그 실험은 의도한 것을 검증하지 못한 것입니다.
- **B4는 무음 프레임에서도 패치 코드를 더합니다** (의도된 예외). 위치 기저는 소리 유무와
  무관하게 동일해야 하기 때문입니다. 시각 스트림의 엄격한 무음 불변성이 더 중요하면
  `shared_grid_code=False`로 끄세요. 게이트 0에서 no-op인 것은 B4도 동일합니다.
- 기존 체크포인트에서 이어받을 땐 `gate_init=0.0`으로 시작해 no-op에서 출발하세요.
  base weight에서 새로 학습하면 `0.1` 정도가 무난합니다.
- `b2_sigma`도 함께 보세요. σ가 계속 커지면 위치 정보를 못 쓰고 전역 바이어스로 퇴화하는
  중이라는 뜻입니다.
- 렌더가 정사각형이 아니게 바뀌면 `uv_after_resize_with_pad`로 (u,v)를 letterbox 보정한
  뒤 넘겨야 합니다. 현재 VLABench 평가 경로는 224×224로 직접 렌더하므로 항등입니다.

---

## 8. 검증되지 않은 것 (솔직히)

- 단위 테스트 21개는 기하·마스킹·게이팅·형상·gradient 흐름을 검증합니다
  (`python -m pytest src/audio_visual_bridge/tests/test_bridges.py -q`).
- **실제 pi0 체크포인트에 붙여 학습해 본 적은 아직 없습니다.** 위 표의 "약한 축을 살린다"는
  것은 설계 의도이지 측정된 결과가 아닙니다.
- B3는 `attach_to_pi0`로는 적용되지 않습니다 (어텐션 마스크에 접근 불가). 경로 2에서만
  동작하며, 부착 시 경고를 냅니다.
