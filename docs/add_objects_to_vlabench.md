# VLABench에 새 오브젝트 추가하기

이 문서는 LIBERO에서 쓰던 `glb/fbx -> obj -> obj2mjcf -> XML 후처리 -> 등록` 흐름을 VLABench 구조에 맞게 바꾼 절차다.

## 핵심 차이

LIBERO와 VLABench의 차이는 주로 후처리와 등록 방식에 있다.

| 항목 | LIBERO | VLABench |
| --- | --- | --- |
| asset 위치 | `assets/stable_hope_objects/<object>/` | `VLABench/assets/obj/meshes/<category>/<object>/` |
| XML 필수 body | 보통 inner body `name="object"` 필요 | 특정 body 이름보다 entity XML 자체를 로드 |
| site annotation | `bottom_site`, `top_site`, `horizontal_radius_site` | `grasppoint`, `placepoint`, `keypoint` 또는 site `group` |
| 등록 위치 | `custom_objects.py` class 등록 | `VLABench/configs/constant.py`의 `name2class_xml` |
| task 사용 | object class 직접 사용 | task config의 `seen_object` / `unseen_object` key로 선택 |

VLABench에서 가장 중요한 연결 구조는 다음이다.

- `VLABench/configs/constant.py`
  - `name2class_xml`: object name -> `[Entity class, xml path/list]`
  - `get_object_list(...)`: 특정 폴더 아래의 `.xml`을 재귀적으로 수집
- `VLABench/tasks/config_manager.py`
  - task의 `seen_object`, `unseen_object`에서 object key를 뽑고 `name2class_xml`로 XML/class를 찾음
- `VLABench/tasks/components/entity.py`
  - `group=2`: place site
  - `group=3`: key site
  - `group=4`: grasp site

## Step 1. 모델 다운로드

모델 소스는 LIBERO와 동일하게 쓸 수 있다.

### Objaverse

```bash
pip install objaverse
python3 - <<'PY'
import objaverse

anns = objaverse.load_annotations()
hits = {
    uid: ann
    for uid, ann in anns.items()
    if "smartphone" in ann["name"].lower()
}
uids = list(hits.keys())[:5]
print(uids)
objaverse.load_objects(uids)
PY
```

### Google Scanned Objects

```bash
huggingface-cli download google-research-datasets/scanned-objects-with-articulations
```

### Sketchfab

무료/CC 라이선스 모델을 `.glb`, `.fbx`, `.obj` 등으로 다운로드한다. 라이선스 표기는 별도로 기록해두는 편이 좋다.

## Step 2. Blender로 `.obj` 변환

`.glb` 예시:

```bash
blender --background --python-expr "
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='/path/to/model.glb')

# 필요하면 단위 보정. 예: mm -> m
for obj in bpy.data.objects:
    obj.scale = (0.001, 0.001, 0.001)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

bpy.ops.export_scene.obj(
    filepath='/path/to/output/model.obj',
    use_materials=True,
    use_uvs=True,
)
"
```

`.fbx`는 import 부분만 바꾼다.

```python
bpy.ops.import_scene.fbx(filepath="/path/to/model.fbx")
```

## Step 3. obj2mjcf로 MJCF 생성

```bash
pip install obj2mjcf

obj2mjcf \
  --obj-dir /path/to/smartphone \
  --save-mjcf \
  --compile-model \
  --decompose
```

단일 OBJ면:

```bash
obj2mjcf \
  --obj-path /path/to/phone.obj \
  --save-mjcf \
  --compile-model \
  --decompose
```

대략 다음 산출물이 생긴다.

```text
model/
  model.xml
  model.obj
  model_collision_*.obj
  texture_map.png
```

## Step 4. VLABench asset tree로 배치

VLABench는 보통 이 경로 아래의 XML을 쓴다.

```text
VLABench/assets/obj/meshes/<category>/<object>/<variant>/
```

예를 들어 스마트폰을 tool category로 넣는다면:

```text
VLABench/assets/obj/meshes/tools/smartphone/smartphone_0/
  model.xml
  model.obj
  model_collision_0.obj
  model_collision_1.obj
  texture_map.png
```

XML 안의 mesh file 경로는 `model.xml` 기준 상대경로가 가장 관리하기 쉽다.

```xml
<mesh name="model" file="model.obj" scale="1 1 1"/>
<mesh name="model_collision_0" file="model_collision_0.obj" scale="1 1 1"/>
```

기존 VLABench asset처럼 `assets/...` 하위 폴더를 따로 둬도 된다. 그 경우 XML의 `file` 경로와 실제 파일 배치를 맞춰야 한다.

## Step 5. XML에 VLABench annotation 추가

LIBERO의 `bottom_site`, `top_site`, `horizontal_radius_site`는 VLABench 일반 grasp object에는 필수가 아니다. 대신 VLABench는 site의 `group`을 사용한다.

### grasp 가능한 일반 오브젝트

최소한 grasp point 하나를 넣는 것을 권장한다.

```xml
<mujoco model="smartphone">
  <compiler angle="radian"/>

  <default>
    <default class="visual">
      <geom group="2" type="mesh" contype="0" conaffinity="0" mass="0.01"/>
    </default>
    <default class="collision">
      <geom group="3" type="mesh" solref="0.001 2" solimp="0.998 0.998 0.001" mass="0"/>
    </default>
    <default class="grasppoint">
      <site type="sphere" size="0.01" group="4" rgba="0 0 1 1"/>
    </default>
    <default class="keypoint">
      <site type="sphere" size="0.01" group="3" rgba="1 0 0 0"/>
    </default>
    <default class="placepoint">
      <site type="sphere" size="0.01" group="2" rgba="0 0 1 0"/>
    </default>
  </default>

  <asset>
    <mesh name="smartphone_visual" file="model.obj" scale="1 1 1"/>
    <mesh name="smartphone_collision_0" file="model_collision_0.obj" scale="1 1 1"/>
  </asset>

  <worldbody>
    <body name="smartphone_body">
      <geom mesh="smartphone_visual" class="visual"/>
      <geom mesh="smartphone_collision_0" class="collision"/>

      <!-- robot grasp target -->
      <site name="smartphone_grasp" class="grasppoint" pos="0 0 0.03"/>

      <!-- optional semantic/key points -->
      <site name="smartphone_top" class="keypoint" pos="0 0 0.06"/>
      <site name="smartphone_place" class="placepoint" pos="0 0 0.07"/>
    </body>
  </worldbody>
</mujoco>
```

### container/receptacle 오브젝트

컨테이너라면 `placepoint`와 `keypoint`가 중요하다.

- `placepoint`, `group=2`: 물체를 놓을 후보 위치
- `keypoint`, `group=3`: contain 판정에 쓰는 bounding/key point

평평한 접시/트레이류는 기존 LIBERO식 `horizontal_radius_site`가 일부 VLABench container 코드에서 쓰일 수 있다. 특히 `FlatContainer.get_radius()`는 `horizontal_radius_site` 이름을 찾는다. 따라서 flat circular container는 다음을 추가하는 편이 안전하다.

```xml
<site name="horizontal_radius_site" class="keypoint" pos="0.1 0 0"/>
<site name="place_center" class="placepoint" pos="0 0 0.02"/>
```

## Step 6. `name2class_xml`에 등록

등록 파일:

```text
VLABench/configs/constant.py
```

기존 category에 instance만 추가하는 경우:

- 예: 새 apple XML을 `VLABench/assets/obj/meshes/fruit/apple/apple_new/model.xml`에 넣음
- `constant.py`의 `"apple": [components.Fruit, get_object_list(...)]`가 이미 있으면 별도 코드 수정 없이 자동 수집

완전히 새 object key를 만들 경우:

```python
"smartphone": [
    components.CommonGraspedEntity,
    get_object_list(os.path.join(xml_root, "obj/meshes/tools/smartphone")),
],
```

특정 XML 하나만 등록할 수도 있다.

```python
"smartphone": [
    components.CommonGraspedEntity,
    "obj/meshes/tools/smartphone/smartphone_0/model.xml",
],
```

새 class behavior가 필요하면 `CommonGraspedEntity`, `CommonContainer`, `FlatContainer` 등을 상속한 class를 만들고 `@register.add_entity(...)`로 등록한 뒤 `name2class_xml`에서 그 class를 사용한다.

## Step 7. task config에 추가

task에서 랜덤 선택되게 하려면 `VLABench/configs/task_config.json`의 해당 task series에 object key를 추가한다.

예:

```json
"select_tool_series": {
  "task": {
    "asset": {
      "seen_object": ["smartphone"],
      "unseen_object": ["another_phone"]
    }
  }
}
```

기존 task에 넣을 수도 있다.

```json
"select_toy_series": {
  "task": {
    "asset": {
      "seen_object": ["smartphone", "..."],
      "unseen_object": ["..."]
    }
  }
}
```

다만 object category와 task semantics가 맞아야 한다. 예를 들어 `select_fruit`에 smartphone을 넣으면 XML은 로드될 수 있지만 instruction/condition 의미가 어색해진다.

## Step 8. 검증

### XML compile 확인

```bash
python - <<'PY'
from dm_control import mjcf

xml_path = "VLABench/assets/obj/meshes/tools/smartphone/smartphone_0/model.xml"
model = mjcf.from_path(xml_path)
print(model.model)
print("meshes:", len(model.find_all("mesh")))
print("geoms:", len(model.find_all("geom")))
print("sites:", [(s.name, s.group) for s in model.find_all("site")])
PY
```

### MuJoCo viewer로 크기 확인

```bash
python -m mujoco.viewer --mjcf VLABench/assets/obj/meshes/tools/smartphone/smartphone_0/model.xml
```

크기가 이상하면 XML의 `<mesh scale="...">`을 조정한다.

```xml
<mesh name="smartphone_visual" file="model.obj" scale="0.001 0.001 0.001"/>
```

VLABench는 runtime domain randomization에서도 mesh/site scale을 조정할 수 있으므로, 기본 XML scale과 task randomness scale이 중복으로 커지거나 작아지지 않는지 확인한다.

## 자주 나는 문제

### 텍스처가 안 보임

- `.mtl`의 `map_Kd`가 절대경로를 가리키는지 확인
- XML mesh file 경로가 `model.xml` 기준으로 맞는지 확인
- texture png/jpg가 실제 위치에 있는지 확인

### grasp가 이상함

- `group=4` site가 있는지 확인
- `grasppoint` 위치가 실제 잡기 좋은 위치인지 viewer에서 확인
- 너무 얇거나 복잡한 물체는 collision hull이 손가락 contact를 방해할 수 있음

### contain/place condition이 실패함

- container class가 `CommonContainer`인지 `FlatContainer`인지 확인
- `placepoint`는 `group=2`
- `keypoint`는 `group=3`
- flat circular container는 `horizontal_radius_site` 이름이 필요할 수 있음

### task에서 object key를 못 찾음

- `VLABench/configs/constant.py`의 `name2class_xml`에 key가 있는지 확인
- `task_config.json`의 `seen_object` / `unseen_object`에 같은 key를 썼는지 확인
- `get_object_list(...)` 경로 아래에 `.xml` 파일이 실제로 있는지 확인

## 최소 체크리스트

- [ ] `.glb` / `.fbx` / `.obj` 확보
- [ ] Blender 또는 trimesh로 `.obj` 변환
- [ ] `obj2mjcf --compile-model --save-mjcf --decompose` 실행
- [ ] 결과물을 `VLABench/assets/obj/meshes/<category>/<object>/...` 아래 배치
- [ ] XML mesh 경로와 scale 확인
- [ ] grasp object면 `group=4` grasp site 추가
- [ ] container면 `group=2` place site와 `group=3` key site 추가
- [ ] `VLABench/configs/constant.py`의 `name2class_xml`에 등록
- [ ] 필요한 task의 `task_config.json`에 object key 추가
- [ ] `dm_control.mjcf.from_path(...)`로 XML compile 확인
- [ ] MuJoCo viewer 또는 task 실행으로 크기/contact/grasp 확인
