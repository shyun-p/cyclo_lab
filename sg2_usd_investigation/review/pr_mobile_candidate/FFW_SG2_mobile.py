# Copyright (c) 2026, Cyclo Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Explicit mobile SG2 configuration, preserving the fixed pick_place defaults.

The caller must provide a floating-base USD and wheel actuator parameters.
This factory neither repairs a USD nor creates sensors or task actions.
"""
from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .FFW_SG2 import FFW_SG2_CFG


def make_ffw_sg2_mobile_cfg(
    usd_path: str,
    *,
    wheel_steer: ImplicitActuatorCfg,
    wheel_drive: ImplicitActuatorCfg,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> ArticulationCfg:
    """Copy the SG2 cfg and add independent steer/drive actuator groups.

    Supply wheel parameters in Isaac Lab's SI units. They have no hardware
    defaults here. Drive stiffness must be zero for velocity control.
    The USD must already provide wheel colliders and valid mimic constraints.
    Dependent gripper joints follow mimic without a competing position servo.
    """
    if not isinstance(usd_path, str) or not usd_path.strip():
        raise ValueError("A floating-base SG2 USD path is required")
    if wheel_drive.stiffness != 0:
        raise ValueError("Wheel drive stiffness must be zero for velocity control")

    cfg = FFW_SG2_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn.usd_path = usd_path
    cfg.spawn.rigid_props.disable_gravity = False
    cfg.init_state.pos = (0.0, 0.0, 0.02)
    cfg.init_state.joint_pos.update({
        f"{wheel}_wheel_{kind}": 0.0
        for wheel in ("left", "right", "rear") for kind in ("steer", "drive")
    })
    cfg.actuators["wheel_steer"] = deepcopy(wheel_steer)
    cfg.actuators["wheel_steer"].joint_names_expr = [".*_wheel_steer"]
    cfg.actuators["wheel_drive"] = deepcopy(wheel_drive)
    cfg.actuators["wheel_drive"].joint_names_expr = [".*_wheel_drive"]
    cfg.actuators["gripper_slave"] = cfg.actuators["gripper_slave"].replace(
        stiffness=0.0, damping=0.0,
    )
    return cfg
