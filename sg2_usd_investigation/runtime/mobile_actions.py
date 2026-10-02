"""Separate mobile action configuration for an explicitly adapted trial task.

Initialize the original task action mode, then pass its configured ActionsCfg to
make_mobile_actions_cfg. This appends three wheel-speed and three steering-angle
commands to the original 19-dimensional action layout, for 25 dimensions total.

This module only received Python syntax validation locally. It is not an
automatic task wrapper: frame/sensor paths, base reset, observations, policy
inputs, recorded datasets and Mimic action assembly require separate changes.
No Isaac Lab 2.3.0 + Isaac Sim 6.0.1-rc.7 full-task execution is claimed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING

from isaaclab.envs.mdp import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.utils import configclass

from cyclo_lab.manager_based.manipulation.pick_place.config.ffw_sg2.pick_place_env_cfg import ActionsCfg


ORIGINAL_ACTION_NAMES = (
    "arm_l_action",
    "gripper_l_action",
    "arm_r_action",
    "gripper_r_action",
    "lift_action",
    "head_action",
)
MOBILE_ACTION_NAMES = ORIGINAL_ACTION_NAMES + ("wheel_drive_action", "wheel_steer_action")
WHEEL_DRIVE_JOINTS = ("left_wheel_drive", "right_wheel_drive", "rear_wheel_drive")
WHEEL_STEER_JOINTS = ("left_wheel_steer", "right_wheel_steer", "rear_wheel_steer")
ORIGINAL_ACTION_DIM = 19
MOBILE_ACTION_DIM = 25


@configclass
class TrialMobileActionsCfg(ActionsCfg):
    """Retain the six original action terms and append wheel drive/steer terms."""

    wheel_drive_action: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(WHEEL_DRIVE_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    wheel_steer_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(WHEEL_STEER_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


def make_mobile_actions_cfg(base_actions: ActionsCfg) -> TrialMobileActionsCfg:
    """Copy initialized original action terms and add independent wheel targets.

    Expected original modes are record/inference/mimic_ik. Wheel drive commands
    have units rad/s; steer commands have units rad. Body vx/vy/wz commands need
    an additional swerve controller. Changing ActionsCfg alone does not provide it.
    """
    copied_terms = {}
    for name in ORIGINAL_ACTION_NAMES:
        term = getattr(base_actions, name, MISSING)
        if term is MISSING or term is None:
            raise ValueError(f"Initialize the original task action mode first; {name} is unset")
        copied_terms[name] = deepcopy(term)
    return TrialMobileActionsCfg(**copied_terms)


def clamp_gripper_command(command):
    """Clamp a scalar or tensor command to the trial common range [0, 1.0] rad.

    Call this in the action producer on both gripper targets. It is deliberately
    explicit: make_mobile_actions_cfg does not alter the original gripper terms.
    Isaac Lab 2.3 JointPositionActionCfg does not expose a clip parameter.
    """
    tensor_clamp = getattr(command, "clamp", None)
    if tensor_clamp is not None:
        return tensor_clamp(0.0, 1.0)
    return min(max(float(command), 0.0), 1.0)
