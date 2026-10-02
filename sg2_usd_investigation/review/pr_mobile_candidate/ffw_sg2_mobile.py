# Copyright (c) 2026, Cyclo Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Bounded SG2 mobile example with explicitly supplied wheel parameters.

Requires a separately prepared floating-base SG2 USD. Does not run pick_place,
create calibrated sensors, load policies, or communicate with real hardware.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, required=True, help="Floating-base SG2 USD; never saved")
parser.add_argument("--out", type=Path, help="Optional result JSON outside the source tree")
for field in ("steer-stiffness", "steer-damping", "steer-effort", "steer-speed-limit",
              "drive-damping", "drive-effort", "drive-speed-limit"):
    parser.add_argument("--" + field, type=float, required=True)
parser.add_argument("--drive-speed", type=float, default=10.0, help="Example wheel command in rad/s")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.usd = args.usd.resolve(strict=True)
if args.drive_speed <= 0 or args.drive_speed > args.drive_speed_limit:
    parser.error("Drive command must be positive and within the supplied speed limit")
if any(getattr(args, field) <= 0 for field in ("steer_stiffness", "steer_damping", "steer_effort",
                                             "steer_speed_limit", "drive_damping", "drive_effort", "drive_speed_limit")):
    parser.error("Wheel gains and limits must be positive")
app = AppLauncher(args).app

import torch
from pxr import Usd, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG
from cyclo_lab.assets.robots.FFW_SG2_mobile import make_ffw_sg2_mobile_cfg

DT = .005
result = {"status": "starting", "scope": "Isaac Lab mobile scene; not pick_place, sensors, policies or hardware",
          "usd_sha256_before": hashlib.sha256(args.usd.read_bytes()).hexdigest(),
          "wheel_parameters": {k: v for k, v in vars(args).items() if k.startswith(("steer_", "drive_"))},
          "dt_s": DT, "events": []}


def main():
    stage = Usd.Stage.Open(str(args.usd))
    world_anchors = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(prim)
            if joint.GetJointEnabledAttr().Get() is not False:
                if bool(joint.GetBody0Rel().GetTargets()) != bool(joint.GetBody1Rel().GetTargets()):
                    world_anchors.append(str(prim.GetPath()))
    if world_anchors:
        raise ValueError(f"USD contains enabled world fixed joints: {world_anchors}")

    before = FFW_SG2_CFG.to_dict()
    steer = ImplicitActuatorCfg(joint_names_expr=[".*_wheel_steer"], stiffness=args.steer_stiffness,
                               damping=args.steer_damping, effort_limit_sim=args.steer_effort,
                               velocity_limit_sim=args.steer_speed_limit)
    drive = ImplicitActuatorCfg(joint_names_expr=[".*_wheel_drive"], stiffness=0.,
                               damping=args.drive_damping, effort_limit_sim=args.drive_effort,
                               velocity_limit_sim=args.drive_speed_limit)
    mobile = make_ffw_sg2_mobile_cfg(str(args.usd), wheel_steer=steer, wheel_drive=drive)
    result["original_cfg_unchanged"] = before == FFW_SG2_CFG.to_dict()
    result["caller_actuators_copied"] = mobile.actuators["wheel_steer"] is not steer and mobile.actuators["wheel_drive"] is not drive
    @configclass
    class MobileSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = mobile

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device, use_fabric=False))
    # Local floor avoids downloading a ground-plane asset. Its friction is an
    # explicit example setting and does not represent a measured tire/surface.
    floor = sim_utils.CuboidCfg(size=(200., 200., .1),
                               collision_props=sim_utils.CollisionPropertiesCfg(),
                               physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.))
    floor.func("/World/Ground", floor, translation=(0., 0., -.05))
    scene = InteractiveScene(MobileSceneCfg(num_envs=1, env_spacing=3.))
    sim.reset()
    robot = scene["robot"]
    positions = robot.data.default_joint_pos.clone()
    velocities = torch.zeros_like(positions)
    names = robot.joint_names
    drive_ids = [names.index(f"{side}_wheel_drive") for side in ("left", "right", "rear")]
    steer_ids = [names.index(f"{side}_wheel_steer") for side in ("left", "right", "rear")]

    def snapshot():
        q = robot.data.joint_pos[0]
        return {"root_position_m": robot.data.root_pos_w[0].tolist(),
                "gripper_rad": {s: [float(q[names.index(f"gripper_{s}_joint{k}")]) for k in range(1, 5)] for s in ("l", "r")}}

    def advance(seconds, wheel_speed=0., steer_angle=0., gripper=0.):
        positions[:, steer_ids] = steer_angle
        for side in ("l", "r"):
            positions[:, names.index(f"gripper_{side}_joint1")] = min(max(gripper, 0.), 1.)
        for _ in range(round(seconds / DT)):
            aligned = bool(torch.max(torch.abs(robot.data.joint_pos[:, steer_ids] - steer_angle)) < .18)
            velocities[:, drive_ids] = wheel_speed if aligned else 0.
            robot.set_joint_position_target(positions)
            robot.set_joint_velocity_target(velocities)
            scene.write_data_to_sim()
            sim.step(render=not args.headless)
            scene.update(DT)
        state = snapshot()
        result["events"].append(state)
        return state

    initial = advance(2.)
    forward = advance(3., args.drive_speed)
    stopped = advance(.5)
    turned = advance(.6, steer_angle=math.pi / 2)
    lateral = advance(2., args.drive_speed, math.pi / 2)
    closed = advance(1.5, steer_angle=math.pi / 2, gripper=.8)
    opened = advance(1.5, steer_angle=math.pi / 2)
    forward_delta = [b-a for a,b in zip(initial["root_position_m"], forward["root_position_m"])]
    lateral_delta = [b-a for a,b in zip(turned["root_position_m"], lateral["root_position_m"])]
    follow_error = max(abs(q[k]-q[0]) for q in closed["gripper_rad"].values() for k in (1, 2, 3))
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()
    result.update(forward_delta_m=forward_delta, lateral_delta_m=lateral_delta,
                  gripper_follow_error_rad=follow_error, gripper_open_rad=opened["gripper_rad"],
                  reset_root_position_m=robot.data.root_pos_w[0].tolist())
    result["checks"] = {
        "original_cfg_unchanged": result["original_cfg_unchanged"],
        "caller_actuators_copied": result["caller_actuators_copied"],
        "forward_displacement": forward_delta[0] > .5,
        "lateral_displacement": lateral_delta[1] > .5,
        "gripper_reaches_command": all(abs(q[0]-.8) < .02 for q in closed["gripper_rad"].values()),
        "gripper_follows": follow_error < .02,
        "gripper_opens": all(max(abs(v) for v in q) < .02 for q in opened["gripper_rad"].values()),
        "finite_state": all(math.isfinite(v) for event in result["events"] for v in event["root_position_m"]),
    }
    result["pass"] = all(result["checks"].values())
    result["status"] = "passed" if result["pass"] else "failed"


try:
    main()
except Exception:
    result.update(status="failed", **{"pass": False}, traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    result["usd_unchanged"] = result["usd_sha256_before"] == hashlib.sha256(args.usd.read_bytes()).hexdigest()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("SG2_MOBILE_EXAMPLE:", result["status"], "usd_unchanged=", result["usd_unchanged"], flush=True)
    app.close()
