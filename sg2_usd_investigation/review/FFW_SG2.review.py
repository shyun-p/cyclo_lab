# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR

FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        # SG2_REVIEW_BEGIN isolated-usd
        # [검토 제안] 원본 파일과 공식 경로를 보존하고 복사한 실험 USD만 사용합니다.
        # 실험 파일: /home/robotis/workspaces/hx5_isaac/cyclo_lab/sg2_usd_investigation/runtime/FFW_SG2_trial.usd
        # usd_path=os.environ["SG2_TRIAL_USD"],
        # 위 줄을 실제 적용하는 별도 실행 파일에는 import os가 필요합니다.
        # 검토본은 모든 제안이 주석이므로 현재 원본과 동일하게 실행됩니다.
        # SG2_REVIEW_END
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            # SG2_REVIEW_BEGIN gravity
            # [검토 제안] world 고정이 없는 주행 실험에서는 중력을 켭니다.
            # disable_gravity=False,
            # 중력만 켜서는 해결되지 않습니다. 고정 조인트, 바퀴 접촉, 액추에이터와 명령도 확인해야 합니다.
            # SG2_REVIEW_END
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # SG2_REVIEW_BEGIN self-collision
            # [검토 설명] v4 후보는 자기충돌을 유지하고 겹치는 5쌍만 제외합니다.
            # base_link ↔ 세 wheel_drive_link, arm_base_link ↔ lift_link/head_link2.
            # v4의 충돌 제외는 근사 충돌체 문제에 대한 실험 조치입니다. 실제 접촉 정확도는 별도 확인 대상입니다.
            # SG2_REVIEW_END
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # SG2_REVIEW_BEGIN trial-start-height
        # [검토 제안] 단독 접촉 시험에서 바닥보다 2 cm 높게 놓고 정착 과정을 관찰합니다.
        # pos=(0.0, 0.0, 0.02),
        # 이 값은 접촉 정착 확인용 초기 배치이며 실제 로봇 치수나 task 배치를 바꾸는 사양값이 아닙니다.
        # SG2_REVIEW_END
        joint_pos={
            # # Swerve base joints
            # "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
            # "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
            # "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,
            # SG2_REVIEW_BEGIN wheel-initial-state
            # [검토 제안] 별도 cfg에서 바퀴 초기값 6개를 활성화합니다.
            # "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
            # "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
            # "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,
            # SG2_REVIEW_END

            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

            # Head joints
            "head_joint1": 0.0,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={
        # Actuators for swerve base
        # "base": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "left_wheel_drive", "left_wheel_steer",
        #         "right_wheel_drive", "right_wheel_steer",
        #         "rear_wheel_drive", "rear_wheel_steer",
        #     ],
        #     velocity_limit_sim=30.0,
        #     effort_limit_sim=100000.0,
        #     stiffness=10000.0,
        #     damping=100.0,
        # ),
        # SG2_REVIEW_BEGIN separate-wheel-actuators
        # [검토 제안] 조향 위치 제어와 구동 속도 제어를 분리합니다.
        # 아래 힘/속도/게인은 동작 확인용 임시값입니다. ROBOTIS 모터 사양 확인 후 확정해야 합니다.
        # "wheel_steer": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_wheel_steer"],
        #     velocity_limit_sim=30.0, effort_limit_sim=1000.0,
        #     stiffness=10000.0, damping=100.0,
        # ),
        # "wheel_drive": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_wheel_drive"],
        #     velocity_limit_sim=30.0, effort_limit_sim=100.0,
        #     stiffness=0.0, damping=100.0,
        # ),
        # 구동 바퀴는 위치를 고정하지 않고 속도 목표를 받습니다.
        # SG2_REVIEW_END

        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42" : ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Actuators for grippers
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),
        # SG2_REVIEW_BEGIN passive-gripper-trial
        # [검토 제안] mimic 구속이 유효한 실험 USD에서는 종속 조인트의 PD를 제거해 비교합니다.
        # "gripper_slave": ImplicitActuatorCfg(
        #     joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
        #     effort_limit_sim=20.0, stiffness=0.0, damping=0.0,
        # ),
        # 이전 5.1 old_cfg.log에서는 원본도 연동 오차 0입니다.
        # 현재 6.0.1 standalone 비교에서는 원본의 종속 PD 유지 시 연동 오차 0.226 rad,
        # 종속 stiffness/damping만 0으로 바꾸면 2.09e-6 rad로 개선됐습니다.
        # 종속 0 rad PD 목표와 mimic의 경합을 피하는 제안입니다. 전체 Isaac Lab/실물 동일성은 별도 검증 대상입니다.
        # 최신 URDF의 j1/j3 최대 1.1 rad와 j2/j4 최대 1.0 rad가 다릅니다.
        # v4 변환 USD의 종속 제한 [-0.22, 1.32]도 URDF와 다릅니다.
        # 실험용 USD만 모든 gripper joint 범위를 공통 [0, 1.0] rad로 맞추고 명령도 동일하게 제한합니다.
        # 이 조치는 제한 충돌을 피하는 검토값이며 실물의 개폐각, 폭, 힘을 인증하지 않습니다.
        # SG2_REVIEW_END

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)
