"""Isolated Isaac Lab cfg implementing the adjacent commented review proposals.

Load this module only after Isaac Sim/AppLauncher has initialized Isaac Lab.
The official FFW_SG2_CFG is deep-copied; no official file or cfg object is edited.
The USD defaults to FFW_SG2_trial.usd beside this file, or SG2_TRIAL_USD.

This cfg has only received Python syntax validation locally. The independent
PhysX USD probe is separate evidence; it does not validate this Isaac Lab cfg or
the complete pick_place task on Isaac Sim 6.0.1-rc.7 / Isaac Lab 2.3.0.

Wheel limits/gains are provisional simulation values, pending ROBOTIS specs.
Lift and arm gains remain the official values; lift sag under gravity can remain.
Trial gripper limits [0, 1.0] rad are authored in the isolated trial USD. Clamp
produced targets separately; configuring an actuator does not clamp task inputs.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG


WHEEL_NAMES = ("left", "right", "rear")
GRIPPER_TRIAL_LIMIT_RAD = (0.0, 1.0)


def make_trial_cfg(
    prim_path: str = "/World/Robot",
    usd_path: str | os.PathLike[str] | None = None,
) -> ArticulationCfg:
    """Return a new floating-base SG2 cfg referencing only the trial USD.

    This does not update task frame paths, create sensors, add actions, change
    observations, or adapt policy/dataset dimensions. Review those separately.
    """
    candidate = (
        usd_path
        if usd_path is not None
        else os.environ.get("SG2_TRIAL_USD", str(Path(__file__).with_name("FFW_SG2_trial.usd")))
    )
    resolved_usd = Path(candidate).expanduser().resolve()
    if not resolved_usd.is_file():
        raise FileNotFoundError(f"Isolated SG2 trial USD does not exist: {resolved_usd}")

    cfg = deepcopy(FFW_SG2_CFG)
    cfg.prim_path = prim_path
    cfg.spawn.usd_path = str(resolved_usd)
    cfg.spawn.rigid_props.disable_gravity = False
    cfg.spawn.articulation_props.enabled_self_collisions = True
    cfg.init_state.pos = (0.0, 0.0, 0.02)
    cfg.init_state.joint_pos.update(
        {f"{wheel}_wheel_{kind}": 0.0 for wheel in WHEEL_NAMES for kind in ("steer", "drive")}
    )

    # 조향 위치 제어; 검토용 임시 제한과 게인입니다.
    cfg.actuators["wheel_steer"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_wheel_steer"],
        velocity_limit_sim=30.0,
        effort_limit_sim=1000.0,
        stiffness=10000.0,
        damping=100.0,
    )
    # 구동 속도 제어: stiffness=0으로 바퀴 회전각을 고정하지 않습니다.
    cfg.actuators["wheel_drive"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_wheel_drive"],
        velocity_limit_sim=30.0,
        effort_limit_sim=100.0,
        stiffness=0.0,
        damping=100.0,
    )
    # trial USD의 유효한 mimic 구속이 종속 조인트를 구동합니다.
    # 6.0.1 standalone 원본 비교에서도 종속 PD=0으로 연동 오차가 개선됐습니다.
    # 실제 Isaac Lab task/실물 동일성은 별도 검증 대상입니다.
    cfg.actuators["gripper_slave"] = ImplicitActuatorCfg(
        joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
        effort_limit_sim=20.0,
        stiffness=0.0,
        damping=0.0,
    )
    return cfg
