"""Assemble a Korean report from retained evidence; no simulation or source edit."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
R = HERE / "results"
physics = json.loads((R / "physics_final_601.json").read_text())
sensors = json.loads((R / "sensors_wrist_601/sensor_result.json").read_text())
static = json.loads((R / "static_final_audit.json").read_text())
preservation = json.loads((R / "original_preservation.json").read_text())
original = json.loads((R / "physics_original_motor_601.json").read_text())
off = json.loads((R / "physics_original_gripper_off_601.json").read_text())
passive_path = R / "physics_original_gripper_passive_601.json"
passive = json.loads(passive_path.read_text()) if passive_path.exists() else None
candidate = HERE / "runtime/FFW_SG2_trial.usd"
sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
assert physics["passed"] and sensors["pass"]
assert physics["usd_sha256"] == sensors["candidate_sha256_before"] == static["sources"]["candidate"]["sha256"] == sha
assert all(preservation["original_hash_matches"].values())
assert preservation["head_unchanged"] and preservation["git_diff_returncode"] == preservation["git_diff_cached_returncode"] == 0
assert static["candidate_vs_current_urdf"]["match_counts"] == {"mass_match":39,"com_match":39,"full_tensor_match":39,"world_com_zero_q_match":39}
fw, side = physics["forward"]["delta_m"], physics["lateral"]["delta_m"]
follow = max(t["max_follow_error_rad"] for t in physics["gripper_tests"])
source_follow = original["gripper_tests"][0]["max_follow_error_rad"]
off_follow = off["gripper_tests"][0]["max_follow_error_rad"]
passive_note = "종속 관절 PD만 끈 원본 비교 실험은 아직 결과 파일이 없습니다."
if passive:
    positions=passive["gripper_tests"][0]["positions_rad"]
    err=passive["gripper_tests"][0]["max_follow_error_rad"]
    passive_note=f"종속 PD만 0으로 바꾼 원본: 0.8 rad 명령 master 좌/우={positions['l'][0]:.6f}/{positions['r'][0]:.6f} rad, 연동 오차={err:.6f} rad."

report=f"""AI Worker SG2 USD 조사 및 수정 후보 검증 보고서
작성 기준: 2026-10-02 / 공식 파일은 읽기만 함

결과
공식 SG2를 그대로 모바일 로봇으로 사용하기에는 고정 베이스, 중력/모터/행동 설정이 맞지 않습니다.
별도 USD와 코드 후보를 만들고 설치된 Isaac Sim {physics['version']}에서 검증했습니다.
최종 후보의 물리 검사 {len(physics['checks'])}개와 카메라 4개(머리2·손목2)·LiDAR 2개의 거리 검사를 통과했습니다.
실물의 질량 분포, 모터 사양, 센서 보정값, 그리퍼의 실물 동작 일치는 아직 확정하지 않았습니다.

보존 범위
검토 기준 HEAD: f4c0470a5e0af54a18327cf96967e8716d64dbc0 (로컬 upstream/main과 일치).
실시간 원격 최신 여부를 확인하거나 fetch하지 않았습니다. 시작 때부터 origin/main보다 8커밋 앞선 상태였습니다.
원본 FFW_SG2.usd와 FFW_SG2.py의 SHA256 유지, git diff와 staged diff 없음, HEAD 유지.
추가된 파일은 이 checkout의 미추적 sg2_usd_investigation/에만 있습니다.
기존 형제 폴더 ../sg2_usd_investigation/의 조사 자료는 읽기만 했습니다.
commit, push, PR 생성, 공식 GitHub 수정은 수행하지 않았습니다. 기존 upstream push 설정은 no_push입니다.

1. 휠베이스 동작 이상
확인된 원인/사용 조건:
  - 공식 USD world FixedJoint가 베이스를 월드에 고정합니다.
  - 공식 USD 좌우 바퀴의 Mesh 충돌체가 비활성화되어 있습니다. 원본에 Cylinder 바퀴는 없습니다.
  - FFW_SG2.py는 disable_gravity=True이고 바퀴 액추에이터가 주석 처리되어 있습니다.
  - pick_place ActionsCfg에는 팔·그리퍼·리프트·머리 행동만 있으며 바퀴 행동이 없습니다.
이 구성은 고정된 pick_place 작업용이라는 점을 고려해야 하며, 모바일 사용에는 별도 구성이 필요합니다.
원본에 시험 모터를 추가한 6.0.1 장면에서도 직진·옆 이동은 모두 0m로 고정 상태가 재현됐습니다.

별도 수정:
  - 기존 v4 재생성 자료를 복사해 floating base와 최신 URDF 링크 구조를 사용했습니다.
  - 중력을 켜고 조향 위치 제어 / 바퀴 속도 제어를 분리했습니다.
  - 바퀴 충돌은 반지름 0.0865m, 폭 0.05m의 Cylinder로 지정했습니다.
  - v4의 잘못된 Xform CollisionAPI를 실제 Mesh에 옮겨 표준 USD Physics parser가 인식하게 했습니다.
    정적 parser의 오류만으로 이전 PhysX 런타임에서 모든 충돌이 없었다고 단정하지 않습니다.
  - 기존 v4의 자기충돌 제외 5쌍을 유지했습니다. 이는 충돌 근사 모델에 대한 실험 조치입니다.
  - 바퀴 행동 6개를 추가하는 별도 factory와 원문 아래 주석 제안을 만들었습니다.

검증 결과 (물리 CPU, dt=0.005s, 중력 ON, 자기충돌 ON):
  직진 3s: x={fw[0]:.6f}m, 횡오차 y={fw[1]:.6f}m (무미끄럼 기대 2.595m).
  옆 이동 2s: y={side[1]:.6f}m, 횡오차 x={side[0]:.6f}m (무미끄럼 기대 1.730m).
  정지 후 바퀴 중심 높이 약 0.0865m, CoM 지지 삼각형 여유 {physics['settle']['com_support_margin_m']:.6f}m.
  지지 삼각형은 바퀴 중심의 근사이며, 바퀴별 하중/접촉력 전체 검증을 의미하지 않습니다.
  전체 시험 중 최대 기울기 {physics['max_tilt_deg']:.6f}도.
초기 convex 후보는 횡오차 0.107m로 0.1m 기준을 실패했습니다. 판정 기준을 완화하지 않았습니다.
이전 5.1 재생성 v2의 원통 침하와 현재 공식 원본의 문제는 구분해야 합니다.
현재 6.0.1의 최종 원통 시험에서는 침하가 재현되지 않았습니다.
모터 게인·힘·속도 제한은 임시 실험값입니다. ROBOTIS 모터 사양 및 주행 제어 튜닝은 남아 있습니다.

2. 질량 분포 / 관성 / 무게중심
CoM(질량 중심)과 관성 텐서는 서로 다른 물리량입니다.
원본 CoM은 2025-06-23 ba75b8d URDF와 약 2e-8m 이내로 일치합니다.
2026-01-07 24a10b7의 관성 값 변경이 CoM 이상을 만들었다고 단정할 수 없습니다.
저장된 USD 전후 및 Git 기록은 arm_base_link/팔 관성의 1e-6 변경과 world 관성 (3,3,3)을 확인해 줍니다.
수동 편집인지, 어떤 변환 옵션으로 바뀌었는지의 작업 의도는 기록만으로 확정하지 않았습니다.
최신 URDF에는 '직진 yaw 편향을 줄이기 위해 base inertia를 중심으로 옮긴다'는 주석과
base CoM을 (0,0,0)으로 지정한 모델링 변경이 있습니다. 최신 값도 실측 정답이라는 근거는 없습니다.

별도 수정 및 검증:
  - 기준 URDF: sg2_50f368f.urdf와 동일한 현재 로컬 URDF, SHA256 c5ec27133805e2ec1bdbd3c290f563dda0835cf2e22224b6e29a5f0216dbf9eb.
  - 이전 v4는 left LiDAR CoM이 미기재되고 base 관성 비대각 성분이 누락되어 엄밀한 1:1 일치가 아니었습니다.
  - 새 후보는 39개 링크의 질량·로컬 CoM·전체 3x3 관성 텐서·영점 자세 월드 CoM을 모두 보정했습니다.
  - 관성 회전 및 고유축을 함께 비교했고 39/39개가 float32 오차 허용 범위에서 통과했습니다.
  - 최대 관성 행렬 오차 6.91e-8 kg·m², 최대 링크 CoM 오차 1.18e-8m.
  - 질량: 원본 94.223853kg, 현재 URDF 94.423857kg. 최신 LiDAR 두 링크가 각 0.1kg입니다.
  - 영점 자세 전체 CoM[m]: 원본 (-0.046206, 0.003235, 0.595578), 후보 (-0.028897, 0.000720, 0.520974).
원본 metersPerUnit=0.01인데 기하 좌표는 미터 값처럼 작성되어 있습니다.
이번 비교/실험은 1m 단위 장면으로 참조해 수행했으며 원본을 무조건 1/100 축소하지 않았습니다.
최종 후보 metadata는 metersPerUnit=1입니다. 단독으로 USD를 열 때의 단위 해석 차이는 별도 이식성 문제입니다.
현재 URDF의 그리퍼 관성 1e-6 링크 10개 등 실물 적합성은 CAD/실측 질량·CoM·관성 자료가 필요합니다.

3. 카메라 / LiDAR 센서 부재
공식 USD에는 외형과 장착 프레임이 있지만 실제 Camera/LiDAR 센서 prim은 없습니다.
다만 공식 pick_place는 실행 중 head RGB CameraCfg를 생성하므로 '환경 전체에 카메라가 없다'는 설명은 부정확합니다.
URDF의 Gazebo LiDAR XML은 Isaac Sim 센서로 자동 전환되지 않습니다.

별도 수정 및 검증:
  - ZED 좌우 optical frame에 RGB + 이상적인 image-plane depth 카메라 2개를 추가했습니다.
  - 양쪽 손목의 공식 camera_left_link/camera_right_link에 D405 RGB + 이상적 depth 카메라 2개를 추가했습니다.
    위치/관절은 ROBOTIS URDF, optical 축은 RealSense 공식 _d405.urdf.xacro의 명목 좌표를 사용합니다.
    명목 FOV 87x58도는 제조사 자료 기준이며, 영상 320x180/30Hz와 렌더러 clip 0.07~2m는 시험 설정입니다.
    두 손목의 기준 물체 깊이: {sensors['camera']['wrist_left']['measured_center_depth_m']:.6f}/{sensors['camera']['wrist_right']['measured_center_depth_m']:.6f}m (기대 0.150m).
    실물의 장착 오차/내부 보정/왜곡/스테레오 depth 오차는 재현하지 않았습니다.
  - 좌우 LiDAR 링크에 실제 OmniLidar 2개와 데이터 읽기 코드를 추가했습니다.
  - RGB 이미지는 양쪽 모두 240x320x3, 기준 물체 깊이는 양쪽 {sensors['camera']['left']['measured_center_depth_m']:.6f}m (기대 2.600m).
  - LiDAR 기준 벽 local x는 좌/우 {sensors['lidar']['l']['measured_wall_plane_local_x_m']:.6f}/{sensors['lidar']['r']['measured_wall_plane_local_x_m']:.6f}m (기대 3.900m), 각 27개 기준 벽 반사점.
  - 방향 초기화 및 GMO CPU 버퍼 취득/좌표 변환을 실제 6.0.1 API에 맞춰 검증했습니다.
    초기 시험 카메라 방향 오류는 새 시험 스크립트의 wrapper 초기화 순서 문제였으며 URDF optical frame 오류가 아닙니다.
  - 시험 장면은 센서 거리 검증을 위해 로봇 물리만 정지시킨 fixture입니다.
  - FFW_SG2_with_sensors.usda는 fixture 정지 설정 없이 동적 로봇을 참조하는 센서 자산입니다.
    정지 fixture 거리 검사와 주행 중 센서 동작은 별개입니다. 추가 GUI 확인은 아래 기록을 참고하세요.
    전체 Isaac Lab task 통합 실행은 검증하지 않았습니다.
머리 카메라 내부 파라미터 및 모든 카메라의 실측 보정·왜곡·스테레오 depth 오차와 LiDAR 제품 사양은 미확인입니다.
이번 LiDAR는 360도/1채널/10Hz generic 시험 모델입니다. URDF Gazebo 설정(270도/720 samples) 및 실물 보정은 별도 적용 대상입니다.
6.0.1에서는 RTX 이미지와 LiDAR 출력이 실제로 성공했습니다. 이전 Docker 5.1 크래시가 해결됐다는 의미는 아닙니다.

4. 그리퍼 동작 불일치
원래 이슈 링크나 실물 증상이 없어서 열림 폭·기구 방향·힘·속도 중 어떤 불일치인지 아직 확정할 수 없습니다.
이전 5.1 로그는 원본의 0.8 rad 연동 성공을 기록했지만,
현재 6.0.1 CPU standalone 장면의 원본은 연동 오차 {source_follow:.6f} rad를 보였습니다.
자기충돌만 끈 원본에서도 오차 {off_follow:.6f} rad로 변화가 없어 자기충돌 단독 원인이라는 가설은 지지되지 않았습니다.
{passive_note}
이 비교는 종속 관절의 0rad PD 목표와 mimic의 경합이 현재 시험의 연동 오류에 기여함을 보여 줍니다.
따라서 종속 stiffness/damping을 0으로 두는 수정은 현재 6.0.1 시험에서 근거가 있습니다.
이 비교는 Isaac Lab 전체 실행이 아닙니다. slave DriveAPI 저작 경로 및 속도 상한 처리 등이 실제 Lab과 달라
5.1 vs 6.0.1의 순수 버전 차이나 원본 cfg의 단일 원인으로 일반화하지 않았습니다.

별도 수정 및 검증:
  - mimic slave의 PD를 0으로 두고 주 관절 명령과 mimic 구속이 종속 관절을 움직이게 했습니다.
  - URDF j1/j3 상한 1.1rad와 j2/j4 상한 1.0rad의 교집합인 [0,1.0]rad로 시험 USD 제한을 맞췄습니다.
  - action producer에서 clamp하는 별도 함수와 검토 주석을 만들었습니다. clamp가 task에 자동 연결된 것은 아닙니다.
  - 양손 0.8 → 1.0 → 0rad 명령에서 최대 연동 오차 {follow:.6f}rad.
  - PhysX raycast로 손가락 8개와 gripper base 2개의 실제 충돌체를 확인했습니다.
물체 유지/파지력/최대 개구폭/실물 이동 방향 검증은 별도입니다. 연동 성공을 실물 동일성으로 해석하지 않습니다.

코드 적용/검증 범위
review/*.review.py는 공식 원문을 그대로 두고 바꿀 줄/블록 아래에 한국어 주석으로 제안을 넣었습니다.
추가 주석을 제거하면 원본 바이트와 AST가 동일함을 세 파일에서 확인했습니다.
바이너리 USD는 직접 주석 편집할 수 없어 review/FFW_SG2_trial.physics.review.usda에 원본 속성 발췌와 주석 후보를 넣었습니다.
runtime/에는 별도 실행 cfg/actions factory, 최종 USD, 센서 포함 layer가 있습니다.
Isaac Lab cfg/actions는 실제 5.1/Isaac Lab 2.3 컨테이너에서 import/factory/원본 보존/관절 선택을 추가 검증했습니다.
이 확인은 전체 환경 실행이 아닙니다. 로컬 6.0.1 Python에는 torch/gymnasium/isaaclab이 없고,
공식 cyclo_lab/Isaac Lab 2.3.0은 Isaac Sim 5.1.0 조합을 대상으로 합니다.
전체 pick_place/policy/ROS2/학습은 실행하지 않았습니다.
모바일 task 적용 시 robot root/eef/camera 경로, reset, base 관측, 19→25 행동 차원,
Mimic 행동 조립과 기존 정책/데이터셋 호환을 함께 변경해야 합니다. 기존 task에 USD 경로만 바꿔 넣으면 안 됩니다.
Isaac Lab 2.3의 convert_mimic_joints_to_normal_joints 플래그 전달 문제는 해당 버전 소스에서 확인했으며,
6.0 importer의 NewtonMimicAPI 등에 옛 PhysX 후처리를 그대로 적용하면 안 됩니다.
최종 후보는 5.1 v4를 출발점으로 보정했으며 6.0.1 importer로 새로 URDF를 변환한 결과는 아닙니다.

최종 후보 SHA256: {sha}
이 해시는 최종 물리 검사, 센서 검사, 정적 감사 결과에 동일하게 기록되어 있습니다.

근거 파일 (이 폴더 기준)
  review/FFW_SG2.review.py
  review/pick_place_env_cfg.review.py
  review/joint_pos_env_cfg.review.py
  review/FFW_SG2_trial.physics.review.usda
  runtime/sensors.review.usda
  results/static_final_audit.json 및 .table.txt
  results/physics_final_601.json
  results/sensors_wrist_601/sensor_result.json 및 RGB PNG/depth/point NPY
  results/physics_original_motor_601.json / physics_original_gripper_off_601.json
  results/physics_original_gripper_passive_601.json
  results/annotation_ast_check.json / original_preservation.json
실패한 초기 convex/센서 시험도 results/에 보존했습니다. 종료 코드만으로 성공을 판단하지 않았습니다.

외부 기술 근거
  RealSense D405 optical 좌표: https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d405.urdf.xacro
  RealSense D405 명목 FOV/작동 범위: https://www.realsenseai.com/products/d405-series/
  NVIDIA 6.0.1 RTX LiDAR: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_rtx_lidar.html
  ROBOTIS cyclo_lab 구성: https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/README.md
"""
gui_path = R / "manual_gui_601/session.json"
if gui_path.exists():
    gui = json.loads(gui_path.read_text())
    if gui.get("status") == "demo_complete_controls_ready" and gui.get("assets_unchanged"):
        assert gui["robot_usd_sha256"] == sha and gui["assets_unchanged"]
        events = {event["phase"]: event["snapshot"] for event in gui["events"]}
        gui_forward = events["stop"]["root_position_m"][0] - events["forward"]["root_position_m"][0]
        gui_lateral = events["grip_0.8"]["root_position_m"][1] - events["lateral"]["root_position_m"][1]
        hand = events["grip_0"]["gripper_rad"]
        gui_follow = max(abs(finger-master[0]) for master in hand.values() for finger in master[1:])
        report += f"""
추가: GUI에서 직접 조작 가능한 장면 확인 (2026-10-02)
손목 추가 전 scripts/manual_gui_601.py로 로컬 6.0.1-rc.7 화면을 띄웠습니다.
동적 센서 자산을 참조하고 중력/자기충돌을 켠 채 주행, 그리퍼, 카메라 영상, LiDAR 포인트를 함께 확인했습니다.
GUI 시연: 0.35m/s 직진 2.5초 동안 x={gui_forward:.6f}m, 옆 이동 3초 동안 y={gui_lateral:.6f}m.
방향 전환 시 조향이 맞을 때까지 바퀴 속도를 0으로 유지합니다. 이 시연은 11개 물리 회귀 검사와 별도입니다.
그리퍼 0.8rad 명령의 좌/우 마스터={hand['l'][0]:.6f}/{hand['r'][0]:.6f}rad, 종속 연동 최대 오차={gui_follow:.6f}rad.
장착 카메라 실제 시점 영상과 LiDAR 초록 포인트를 함께 표시했으며 센서 읽기 오류={gui['sensor_errors']}입니다.
원본/후보 USD는 저장하지 않았고 후보 SHA256은 위 최종 검증 모델과 같습니다.
조작창: SG2 Manual Control - separate trial. 월드 X/Y 이동 버튼, STOP, 양손 그리퍼 슬라이더,
손 확대, 카메라 영상 창, 무게중심 표시, Reset, Demo once를 제공합니다. 사용법은 RUNBOOK.txt에 있습니다.
근거: results/manual_gui_601/session.json, robot_view.png, head_rgb.png.
first_timing_session.json은 렌더링/물리 step 중복 수정 전 결과이므로 최종 시간/거리 근거로 사용하지 않습니다.
이 확인은 독립 GUI 장면입니다. 전체 pick_place, 실제 로봇 일치, 제품 사양 검증은 여전히 남아 있습니다.
"""
existing_path = R / "manual_existing_wrist_601/session.json"
if existing_path.exists():
    existing = json.loads(existing_path.read_text())
    if existing.get("status") == "demo_complete_controls_ready" and len(existing.get("wrist_camera_paths", {})) == 2:
        assert existing["assets_unchanged"] and not existing["sensor_errors"]
        report += """
추가: 손목 카메라를 포함한 기존 창용 실행기 검증
scripts/open_sg2_in_existing.py의 비동기 초기화/주행/그리퍼 시연을 headless Kit에서 확인했습니다.
머리 영상 창 + 좌우 손목 영상 창 + LiDAR 포인트를 함께 생성했고 각 뷰포트 PNG를 저장했습니다.
근거: results/manual_existing_wrist_601/session.json 및 wrist_left_rgb.png / wrist_right_rgb.png.
새로 실행하면 손목 영상 창이 열립니다. 이전 장면은 자동 교체하지 않으므로 RUNBOOK.txt의 재실행 절차를 따르세요.
이 검증은 사용자의 현재 데스크톱 장면을 직접 조작한 기록과는 구분합니다.
"""
remaining_path = R / "remaining_checks.json"
if remaining_path.exists():
    remaining = json.loads(remaining_path.read_text())
    report += f"""
추가: PR 전 일괄 점검
results/remaining_checks.json에 기존 검증 해시 확인, 실제 5.1 cfg 확인, 6.0.1 의존성 확인, task 경로 검사를 모았습니다.
실제 5.1 cfg/factory 확인={remaining['config_factory_51_pass']}. record/inference/mimic_ik에서 바퀴 관절 선택과 원본 설정 보존을 확인했습니다.
현재 후보에 기존 task의 손끝/카메라 경로를 그대로 적용할 수 있는지={remaining['task_paths']['drop_in_task_paths_pass']}.
arm_base_link, 양쪽 arm_link7, head camera parent의 기존 경로 4곳이 후보 구조와 맞지 않습니다.
전체 Isaac Lab task 실행은5.1과6.0.1 모두 아직 미검증입니다. 6.0.1 런타임에는 torch/gymnasium/isaaclab 의존성도 없습니다.
모바일 base reset, 액션 생성기/정책/데이터셋 연결, LiDAR 사양 적용 및 실제 사양 확인이 남아 있습니다.
문의할 자료와 재확인 절차는 REMAINING_AND_QUESTIONS.ko.txt를 참고하세요.
"""
(HERE / "REPORT.ko.txt").write_text(report, encoding="utf-8")
print(str(HERE / "REPORT.ko.txt"))
