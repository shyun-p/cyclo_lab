#!/usr/bin/env python3
"""Generate annotated SG2 review copies and verify untouched executable originals.

Run with --generate once to create copies. Without that flag this only checks.
All additions are comment-only blocks immediately below affected originals.
The checker compares ASTs AND strips those blocks for exact source-byte equality.
It does not import Isaac Lab, start Isaac Sim, or edit official source files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INVESTIGATION = HERE.parent
REPO = INVESTIGATION.parent
ASSETS = Path("source/cyclo_lab/cyclo_lab/assets/robots")
TASK = Path("source/cyclo_lab/cyclo_lab/manager_based/manipulation/pick_place/config/ffw_sg2")
BEGIN = "# SG2_REVIEW_BEGIN "
END = "# SG2_REVIEW_END"
TRIAL_USD = INVESTIGATION / "runtime/FFW_SG2_trial.usd"
PAIRS = (
    (ASSETS / "FFW_SG2.py", "FFW_SG2.review.py"),
    (TASK / "pick_place_env_cfg.py", "pick_place_env_cfg.review.py"),
    (TASK / "joint_pos_env_cfg.py", "joint_pos_env_cfg.review.py"),
)


def comment_block(label: str, proposal: str, indent: str = "") -> str:
    lines = [indent + BEGIN + label]
    lines.extend(indent + "# " + line if line else indent + "#" for line in proposal.splitlines())
    lines.append(indent + END)
    return "\n".join(lines) + "\n"


def insert_after(source: str, anchor: str, label: str, proposal: str, indent: str = "") -> str:
    if source.count(anchor) != 1:
        raise ValueError(f"Anchor for {label!r} must occur exactly once; got {source.count(anchor)}")
    return source.replace(anchor, anchor + comment_block(label, proposal, indent), 1)


def annotate_robot(source: str) -> str:
    proposals = [
        (
            '        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",\n',
            "isolated-usd",
            "[검토 제안] 원본 파일과 공식 경로를 보존하고 복사한 실험 USD만 사용합니다.\n"
            f'실험 파일: {TRIAL_USD}\n'
            'usd_path=os.environ["SG2_TRIAL_USD"],\n'
            "위 줄을 실제 적용하는 별도 실행 파일에는 import os가 필요합니다.\n"
            "검토본은 모든 제안이 주석이므로 현재 원본과 동일하게 실행됩니다.",
            "        ",
        ),
        (
            "            disable_gravity=True,\n",
            "gravity",
            "[검토 제안] world 고정이 없는 주행 실험에서는 중력을 켭니다.\n"
            "disable_gravity=False,\n"
            "중력만 켜서는 해결되지 않습니다. 고정 조인트, 바퀴 접촉, 액추에이터와 명령도 확인해야 합니다.",
            "            ",
        ),
        (
            "            enabled_self_collisions=True,\n",
            "self-collision",
            "[검토 설명] v4 후보는 자기충돌을 유지하고 겹치는 5쌍만 제외합니다.\n"
            "base_link ↔ 세 wheel_drive_link, arm_base_link ↔ lift_link/head_link2.\n"
            "v4의 충돌 제외는 근사 충돌체 문제에 대한 실험 조치입니다. 실제 접촉 정확도는 별도 확인 대상입니다.",
            "            ",
        ),
        (
            "    init_state=ArticulationCfg.InitialStateCfg(\n",
            "trial-start-height",
            "[검토 제안] 단독 접촉 시험에서 바닥보다 2 cm 높게 놓고 정착 과정을 관찰합니다.\n"
            "pos=(0.0, 0.0, 0.02),\n"
            "이 값은 접촉 정착 확인용 초기 배치이며 실제 로봇 치수나 task 배치를 바꾸는 사양값이 아닙니다.",
            "        ",
        ),
        (
            '            # "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,\n',
            "wheel-initial-state",
            "[검토 제안] 별도 cfg에서 바퀴 초기값 6개를 활성화합니다.\n"
            '"left_wheel_drive": 0.0, "left_wheel_steer": 0.0,\n'
            '"right_wheel_drive": 0.0, "right_wheel_steer": 0.0,\n'
            '"rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,',
            "            ",
        ),
        (
            "        #     damping=100.0,\n        # ),\n",
            "separate-wheel-actuators",
            "[검토 제안] 조향 위치 제어와 구동 속도 제어를 분리합니다.\n"
            "아래 힘/속도/게인은 동작 확인용 임시값입니다. ROBOTIS 모터 사양 확인 후 확정해야 합니다.\n"
            '"wheel_steer": ImplicitActuatorCfg(\n'
            '    joint_names_expr=[".*_wheel_steer"],\n'
            "    velocity_limit_sim=30.0, effort_limit_sim=1000.0,\n"
            "    stiffness=10000.0, damping=100.0,\n"
            "),\n"
            '"wheel_drive": ImplicitActuatorCfg(\n'
            '    joint_names_expr=[".*_wheel_drive"],\n'
            "    velocity_limit_sim=30.0, effort_limit_sim=100.0,\n"
            "    stiffness=0.0, damping=100.0,\n"
            "),\n"
            "구동 바퀴는 위치를 고정하지 않고 속도 목표를 받습니다.",
            "        ",
        ),
        (
            '        "gripper_slave": ImplicitActuatorCfg(\n'
            '            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],\n'
            "            effort_limit_sim=20.0,\n"
            "            stiffness=2.0,\n"
            "            damping=0.5,\n"
            "        ),\n",
            "passive-gripper-trial",
            "[검토 제안] mimic 구속이 유효한 실험 USD에서는 종속 조인트의 PD를 제거해 비교합니다.\n"
            '"gripper_slave": ImplicitActuatorCfg(\n'
            '    joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],\n'
            "    effort_limit_sim=20.0, stiffness=0.0, damping=0.0,\n"
            "),\n"
            "이전 5.1 old_cfg.log에서는 원본도 연동 오차 0입니다.\n"
            "현재 6.0.1 standalone 비교에서는 원본의 종속 PD 유지 시 연동 오차 0.226 rad,\n"
            "종속 stiffness/damping만 0으로 바꾸면 2.09e-6 rad로 개선됐습니다.\n"
            "종속 0 rad PD 목표와 mimic의 경합을 피하는 제안입니다. 전체 Isaac Lab/실물 동일성은 별도 검증 대상입니다.\n"
            "최신 URDF의 j1/j3 최대 1.1 rad와 j2/j4 최대 1.0 rad가 다릅니다.\n"
            "v4 변환 USD의 종속 제한 [-0.22, 1.32]도 URDF와 다릅니다.\n"
            "실험용 USD만 모든 gripper joint 범위를 공통 [0, 1.0] rad로 맞추고 명령도 동일하게 제한합니다.\n"
            "이 조치는 제한 충돌을 피하는 검토값이며 실물의 개폐각, 폭, 힘을 인증하지 않습니다.",
            "        ",
        ),
    ]
    for anchor, label, proposal, indent in proposals:
        source = insert_after(source, anchor, label, proposal, indent)
    return source


def annotate_actions(source: str) -> str:
    source = insert_after(
        source,
        "    head_action: mdp.ActionTermCfg = MISSING\n",
        "optional-mobile-actions",
        "[검토 제안] 이동용으로 분리한 환경에서만 다음 항목을 추가합니다.\n"
        "wheel_drive_action: mdp.ActionTermCfg = MISSING\n"
        "wheel_steer_action: mdp.ActionTermCfg = MISSING\n"
        "원래 19차원 행동에 바퀴 6개가 추가되면 25차원이 됩니다.\n"
        "기존 학습 정책, record/inference 입력, Mimic 행동 조립과 데이터셋을 함께 수정해야 합니다.\n"
        "공식 pick_place의 고정된 행동 순서를 변경하는 제안이므로 이동 시험은 별도 실행기에서 먼저 검증합니다.",
        "    ",
    )
    anchor = (
        '            self.actions.lift_action = mdp.JointPositionActionCfg(\n'
        '                asset_name="robot",\n'
        '                joint_names=["lift_joint"],\n'
        '                scale=1.0,\n'
        '            )\n'
    )
    if source.count(anchor) != 2:
        raise ValueError("Expected exactly two lift action blocks, one per supported action branch")
    proposal = comment_block(
        "wheel-action-branch",
        "[검토 제안] 별도 이동 환경의 record/inference 및 mimic_ik 분기에 다음을 추가합니다.\n"
        "self.actions.wheel_drive_action = mdp.JointVelocityActionCfg(\n"
        '    asset_name="robot", joint_names=["left_wheel_drive", "right_wheel_drive", "rear_wheel_drive"],\n'
        "    scale=1.0, use_default_offset=False, preserve_order=True,\n"
        ")\n"
        "self.actions.wheel_steer_action = mdp.JointPositionActionCfg(\n"
        '    asset_name="robot", joint_names=["left_wheel_steer", "right_wheel_steer", "rear_wheel_steer"],\n'
        "    scale=1.0, use_default_offset=False, preserve_order=True,\n"
        ")\n"
        "구동 명령 단위 rad/s, 조향 명령 단위 rad입니다. 차체 vx/vy/wz를 받으려면 별도 swerve 역기구학이 필요합니다.\n"
        "joint_pos/joint_pos_target 관측에 바퀴를 추가하면 관측 크기도 달라집니다. 정책과 기록 형식을 함께 버전 관리합니다.\n"
        "그리퍼 실험 명령 생산 위치에서는 다음과 같이 공통 [0,1.0] rad를 제한합니다.\n"
        "gripper_action = gripper_action.clamp(0.0, 1.0)\n"
        "위 변수는 명령 생산기에서 정의해야 합니다. Lab 2.3의 JointPositionActionCfg에는 clip 필드가 없습니다.\n"
        "이 검토본은 주석만 추가하며 행동 차원이나 실행 로직을 바꾸지 않습니다.",
        "            ",
    )
    return source.replace(anchor, anchor + proposal)


def annotate_scene(source: str) -> str:
    source = insert_after(
        source,
        '        self.scene.robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")\n',
        "isolated-mobile-config",
        "[검토 제안] 별도 이동 환경에서는 FFW_SG2.review.py의 제안을 적용한 실험 cfg를 사용합니다.\n"
        "self.scene.robot = SG2_TRIAL_CFG.replace(prim_path=\"{ENV_REGEX_NS}/Robot\")\n"
        "SG2_TRIAL_CFG는 공식 파일을 수정하지 않고 별도 모듈에서 생성합니다.\n"
        f"실험 USD 경로: {TRIAL_USD}\n"
        "기존 pick_place의 로봇 배치/테이블/재설정은 고정 베이스를 전제로 하므로 주행 시험은 별도 장면에서 시작합니다.",
        "        ",
    )
    lines = source.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        result.append(line)
        if 'prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/' in line:
            indent = line[:len(line) - len(line.lstrip())]
            proposed = line.strip().replace("/ffw_sg2_follower/", "/")
            label = "v4-prim-path-" + str(len(result))
            text = (
                "[검토 제안] v4/trial USD는 중간 ffw_sg2_follower 계층이 없어 아래 경로를 사용합니다.\n"
                + proposed
                + "\nUSD 교체 시 모든 카메라/eef 경로의 실제 prim 존재를 검사해야 합니다."
            )
            if "/zed/cam_head" in line:
                text += (
                    "\n원본 standalone USD에 Camera prim이 없어도 이 CameraCfg는 task 실행 중 RGB 카메라를 생성합니다."
                    "\n따라서 '카메라 부재'는 USD와 task 실행 상태를 구분해야 합니다. LiDAR는 별도 생성과 출력 검증이 필요합니다."
                    "\n카메라 intrinsics/extrinsics는 현재 설정을 보존하고 ROBOTIS 보정 데이터와 비교할 항목으로 남깁니다."
                )
            result.append(comment_block(label, text, indent))
    return "".join(result)


def strip_proposal_blocks(source: str) -> tuple[str, int]:
    result: list[str] = []
    inside = False
    count = 0
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        trimmed = line.lstrip()
        if trimmed.startswith(BEGIN):
            if inside:
                raise ValueError(f"Nested proposal block at line {number}")
            inside = True
            count += 1
            continue
        if trimmed.rstrip("\r\n") == END:
            if not inside:
                raise ValueError(f"Unmatched proposal end at line {number}")
            inside = False
            continue
        if inside:
            if not trimmed.startswith("#"):
                raise ValueError(f"Proposal contains executable/non-comment line {number}")
        else:
            result.append(line)
    if inside:
        raise ValueError("Unterminated proposal block")
    return "".join(result), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Generate deterministic comment-only review copies")
    args = parser.parse_args()
    annotators = (annotate_robot, annotate_actions, annotate_scene)
    if args.generate:
        (INVESTIGATION / "review").mkdir(parents=True, exist_ok=True)
        for (relative, name), annotate in zip(PAIRS, annotators, strict=True):
            source = (REPO / relative).read_bytes().decode("utf-8")
            (INVESTIGATION / "review" / name).write_bytes(annotate(source).encode("utf-8"))
    checks = []
    for relative, name in PAIRS:
        original_path = REPO / relative
        review_path = INVESTIGATION / "review" / name
        original_bytes = original_path.read_bytes()
        review_bytes = review_path.read_bytes()
        original = original_bytes.decode("utf-8")
        review = review_bytes.decode("utf-8")
        stripped, count = strip_proposal_blocks(review)
        same_ast = ast.dump(ast.parse(original), include_attributes=False) == ast.dump(ast.parse(review), include_attributes=False)
        exact_original = stripped.encode("utf-8") == original_bytes
        checks.append({
            "original": str(original_path),
            "review": str(review_path),
            "proposal_blocks": count,
            "ast_identical": same_ast,
            "original_bytes_identical_after_removing_added_comments": exact_original,
            "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        })
    passed = all(check["ast_identical"] and check["original_bytes_identical_after_removing_added_comments"] and check["proposal_blocks"] > 0 for check in checks)
    report = {"passed": passed, "check_type": "static AST and original byte preservation; no Isaac Sim execution", "checks": checks}
    (INVESTIGATION / "results").mkdir(parents=True, exist_ok=True)
    (INVESTIGATION / "results/annotation_ast_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
