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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    left_eef: FrameTransformerCfg = MISSING
    right_eef: FrameTransformerCfg = MISSING

    brush: AssetBaseCfg = MISSING
    basket: AssetBaseCfg = MISSING
    table: AssetBaseCfg = MISSING
    silicone: AssetBaseCfg = MISSING
    scissors: AssetBaseCfg = MISSING
    driver: AssetBaseCfg = MISSING

    cam_head: CameraCfg = MISSING

    # Background cube for color randomization
    background_cube: AssetBaseCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_l_action: mdp.ActionTermCfg = MISSING
    gripper_l_action: mdp.ActionTermCfg = MISSING
    arm_r_action: mdp.ActionTermCfg = MISSING
    gripper_r_action: mdp.ActionTermCfg = MISSING
    lift_action: mdp.ActionTermCfg = MISSING
    head_action: mdp.ActionTermCfg = MISSING
    # SG2_REVIEW_BEGIN optional-mobile-actions
    # [검토 제안] 이동용으로 분리한 환경에서만 다음 항목을 추가합니다.
    # wheel_drive_action: mdp.ActionTermCfg = MISSING
    # wheel_steer_action: mdp.ActionTermCfg = MISSING
    # 원래 19차원 행동에 바퀴 6개가 추가되면 25차원이 됩니다.
    # 기존 학습 정책, record/inference 입력, Mimic 행동 조립과 데이터셋을 함께 수정해야 합니다.
    # 공식 pick_place의 고정된 행동 순서를 변경하는 제안이므로 이동 시험은 별도 실행기에서 먼저 검증합니다.
    # SG2_REVIEW_END


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": ["arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
                                    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
                                    "head_joint1", "head_joint2", "lift_joint"],
                    "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": ["arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
                                    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
                                    "head_joint1", "head_joint2", "lift_joint"],
                    "asset_name": "robot"},
        )
        left_eef_pose = ObsTerm(func=mdp.eef_pose, params={"eef_cfg": SceneEntityCfg("left_eef"), "robot_cfg": SceneEntityCfg("robot")})
        right_eef_pose = ObsTerm(func=mdp.eef_pose, params={"eef_cfg": SceneEntityCfg("right_eef"), "robot_cfg": SceneEntityCfg("robot")})

        cam_head = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_head"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # Note: object_cfg will be set dynamically based on target_object in __post_init__
        grasp_object = None
        object_in_basket = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Note: success and object_dropped will be set dynamically based on target_object in __post_init__
    success = None
    object_dropped = None


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    # All available objects for randomization
    all_objects: list = ["brush", "driver", "scissors", "pliers", "tooth_brush", "silicone"]
    # Target object configuration
    target_object: str = "brush"  # Options: "silicone", "brush", "scissors", "driver", "pliers", "tooth_brush"
    # Target side configuration: which side to place the target object
    target_side: str = "right"  # Options: "left", "right"

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Determine eef and gripper based on target_side
        if self.target_side == "left":
            eef_name = "left_eef"
            gripper_joint_name = "gripper_l_joint1"
        else:  # right
            eef_name = "right_eef"
            gripper_joint_name = "gripper_r_joint1"

        # Initialize dynamic observations and terminations based on target_object
        self.observations.subtask_terms.grasp_object = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "eef_cfg": SceneEntityCfg(eef_name),
                "object_cfg": SceneEntityCfg(self.target_object),
                "gripper_joint_name": gripper_joint_name,
            },
        )

        self.observations.subtask_terms.object_in_basket = ObsTerm(
            func=mdp.object_in_basket,
            params={
                "object_cfg": SceneEntityCfg(self.target_object),
                "basket_cfg": SceneEntityCfg("basket"),
                "distance_threshold": 0.15,
            },
        )

        self.terminations.success = DoneTerm(
            func=mdp.task_done,
            params={
                "object_cfg": SceneEntityCfg(self.target_object),
                "basket_cfg": SceneEntityCfg("basket"),
                "distance_threshold": 0.15,
            },
        )

        self.terminations.object_dropped = DoneTerm(
            func=mdp.object_dropped,
            params={
                "object_cfg": SceneEntityCfg(self.target_object),
                "velocity_threshold": 2.0,
            },
        )

    def init_action_cfg(self, mode: str):
        print(f"Initializing action configuration for device: {mode}")
        if mode in ['record', 'inference']:
            self.actions.arm_l_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["arm_l_joint[1-7]"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_l_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_l_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.arm_r_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["arm_r_joint[1-7]"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_r_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_r_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.head_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["head_joint1", "head_joint2"],
                scale=1.0,
            )
            self.actions.lift_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["lift_joint"],
                scale=1.0,
            )
            # SG2_REVIEW_BEGIN wheel-action-branch
            # [검토 제안] 별도 이동 환경의 record/inference 및 mimic_ik 분기에 다음을 추가합니다.
            # self.actions.wheel_drive_action = mdp.JointVelocityActionCfg(
            #     asset_name="robot", joint_names=["left_wheel_drive", "right_wheel_drive", "rear_wheel_drive"],
            #     scale=1.0, use_default_offset=False, preserve_order=True,
            # )
            # self.actions.wheel_steer_action = mdp.JointPositionActionCfg(
            #     asset_name="robot", joint_names=["left_wheel_steer", "right_wheel_steer", "rear_wheel_steer"],
            #     scale=1.0, use_default_offset=False, preserve_order=True,
            # )
            # 구동 명령 단위 rad/s, 조향 명령 단위 rad입니다. 차체 vx/vy/wz를 받으려면 별도 swerve 역기구학이 필요합니다.
            # joint_pos/joint_pos_target 관측에 바퀴를 추가하면 관측 크기도 달라집니다. 정책과 기록 형식을 함께 버전 관리합니다.
            # 그리퍼 실험 명령 생산 위치에서는 다음과 같이 공통 [0,1.0] rad를 제한합니다.
            # gripper_action = gripper_action.clamp(0.0, 1.0)
            # 위 변수는 명령 생산기에서 정의해야 합니다. Lab 2.3의 JointPositionActionCfg에는 clip 필드가 없습니다.
            # 이 검토본은 주석만 추가하며 행동 차원이나 실행 로직을 바꾸지 않습니다.
            # SG2_REVIEW_END
        elif mode in ['mimic_ik']:
            self.actions.arm_l_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["arm_l_joint[1-7]"],
                body_name="arm_l_link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.2]),
            )
            self.actions.gripper_l_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_l_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.arm_r_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["arm_r_joint[1-7]"],
                body_name="arm_r_link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.2]),
            )
            self.actions.gripper_r_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_r_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.head_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["head_joint1", "head_joint2"],
                scale=1.0,
            )
            self.actions.lift_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["lift_joint"],
                scale=1.0,
            )
            # SG2_REVIEW_BEGIN wheel-action-branch
            # [검토 제안] 별도 이동 환경의 record/inference 및 mimic_ik 분기에 다음을 추가합니다.
            # self.actions.wheel_drive_action = mdp.JointVelocityActionCfg(
            #     asset_name="robot", joint_names=["left_wheel_drive", "right_wheel_drive", "rear_wheel_drive"],
            #     scale=1.0, use_default_offset=False, preserve_order=True,
            # )
            # self.actions.wheel_steer_action = mdp.JointPositionActionCfg(
            #     asset_name="robot", joint_names=["left_wheel_steer", "right_wheel_steer", "rear_wheel_steer"],
            #     scale=1.0, use_default_offset=False, preserve_order=True,
            # )
            # 구동 명령 단위 rad/s, 조향 명령 단위 rad입니다. 차체 vx/vy/wz를 받으려면 별도 swerve 역기구학이 필요합니다.
            # joint_pos/joint_pos_target 관측에 바퀴를 추가하면 관측 크기도 달라집니다. 정책과 기록 형식을 함께 버전 관리합니다.
            # 그리퍼 실험 명령 생산 위치에서는 다음과 같이 공통 [0,1.0] rad를 제한합니다.
            # gripper_action = gripper_action.clamp(0.0, 1.0)
            # 위 변수는 명령 생산기에서 정의해야 합니다. Lab 2.3의 JointPositionActionCfg에는 clip 필드가 없습니다.
            # 이 검토본은 주석만 추가하며 행동 차원이나 실행 로직을 바꾸지 않습니다.
            # SG2_REVIEW_END
        else:
            raise ValueError(f"Unknown action mode: {mode}")
