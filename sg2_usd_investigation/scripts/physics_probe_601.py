"""Bounded PhysX regression in installed Isaac Sim 6.0.1-rc.7.

This is a standalone USD/controller probe, not a full Isaac Lab task or trained policy.
Original assets are referenced read-only into a 1-metre stage. All mutations are in memory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--case", choices=["original", "original_motor", "trial"], required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--control", choices=["direct", "world"], default="direct")
parser.add_argument("--self-collision", choices=["on", "off"], default="on")
parser.add_argument("--slave-mode", choices=["source", "passive"], default="source")
args, kit_args = parser.parse_known_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"runtime"))
from swerve_controller import wheel_targets

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "disable_viewport_updates": True, "enable_audio": False})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

DT = 1/200
RADIUS = 0.0865
SPEED = 10.0
out = Path(args.out)
result = {"case": args.case, "usd": str(Path(args.usd).resolve()),
          "usd_sha256": hashlib.sha256(Path(args.usd).read_bytes()).hexdigest(),
          "version": Path("/home/robotis/isaacsim/VERSION").read_text().strip(),
          "dt_s": DT, "physics_device": "cpu", "scope": "Standalone PhysX USD and equivalent gains, not Isaac Lab runtime",
          "wheel_control": args.control,
          "self_collision": args.self_collision,
          "slave_mode": "passive" if args.case == "trial" else args.slave_mode,
          "motor_parameters": "Test-only steer kp10000 kd100 effort1000; drive kp0 kd100 effort100 Nm"}


def gains(name):
    # SI gains/effort/velocity from official FFW_SG2.py (except the labelled trial changes).
    if "wheel_" in name:
        if args.case == "original":
            return None
        return (10000, 100, 1000, 30) if name.endswith("steer") else (0, 100, 100, 30)
    if name == "lift_joint":
        return 10000, 100, 1000000, .2
    if name.startswith("head_"):
        return 150, 3, 30, 2
    if name.startswith("arm_"):
        n = int(name[-1])
        if n <= 2:
            return 600, 30, 61.4, 15
        if n <= 6:
            return 600, 20, 31.7, 15
        return 200, 3, 5.1, 6
    if name.startswith("gripper_"):
        if name.endswith("1"):
            return 100, 4, 30, 2.2
        return (0, 0, 20, 2.2) if args.case == "trial" or args.slave_mode == "passive" else (2, .5, 20, 2.2)
    return None


def rotate(v, q):
    # PhysX tensor transform uses xyzw.
    u, w = q[:3], q[3]
    return v + 2*(w*np.cross(u, v)+np.cross(u, np.cross(u, v)))


try:
    world = World(physics_dt=DT, rendering_dt=1/60, stage_units_in_meters=1, backend="numpy", device="cpu")
    world.scene.add_default_ground_plane(static_friction=1., dynamic_friction=1.)
    add_reference_to_stage(args.usd, "/World/Robot")
    stage = omni.usd.get_context().get_stage()
    trial = args.case == "trial"
    if trial:
        x = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robot"))
        x.AddTranslateOp(opSuffix="test_spawn").Set(Gf.Vec3d(0, 0, .02))
    root_paths = []
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith("/World/Robot"):
            continue
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(not trial)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = PhysxSchema.PhysxArticulationAPI.Apply(p)
            art.CreateEnabledSelfCollisionsAttr(args.self_collision == "on")
            art.CreateSolverPositionIterationCountAttr(32)
            art.CreateSolverVelocityIterationCountAttr(1)
            root_paths.append(str(p.GetPath()))
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
            params = gains(p.GetName())
            if params is None:
                continue
            kp, kd, effort, vmax = params
            angular = p.IsA(UsdPhysics.RevoluteJoint)
            drive = UsdPhysics.DriveAPI.Apply(p, "angular" if angular else "linear")
            conversion = math.pi/180 if angular else 1
            drive.CreateStiffnessAttr(kp*conversion)
            drive.CreateDampingAttr(kd*conversion)
            drive.CreateMaxForceAttr(effort)
            drive.CreateTypeAttr("force")
            PhysxSchema.PhysxJointAPI.Apply(p).CreateMaxJointVelocityAttr(vmax/conversion)
    assert len(root_paths) == 1, root_paths
    result["articulation_root"] = root_paths[0]
    robot = world.scene.add(Articulation(root_paths[0], name="sg2"))
    world.reset()
    names = robot.dof_names
    bodies = robot.body_names
    result["joint_names"] = names
    result["body_names"] = bodies
    idx = {name: i for i, name in enumerate(names)}
    bidx = {name: i for i, name in enumerate(bodies)}
    pos = np.zeros((1, len(names)), dtype=np.float32)
    vel = np.zeros_like(pos)
    robot.set_joint_positions(pos)
    robot.set_joint_velocities(vel)
    masses = np.asarray(robot.get_body_masses())[0]
    local_com = np.asarray(robot.get_body_coms()[0])[0]
    result["mass_kg"] = float(masses.sum())
    result["initial_stage_units_m"] = UsdGeom.GetStageMetersPerUnit(stage)

    def snapshot():
        transforms = np.asarray(robot._physics_view.get_link_transforms())[0]
        centers = np.array([t[:3]+rotate(c, t[3:]) for t,c in zip(transforms, local_com)])
        com = (centers*masses[:,None]).sum(0)/masses.sum()
        roots = np.asarray(robot._physics_view.get_root_transforms())[0]
        z_axis = rotate(np.array([0.,0.,1.]), roots[3:])
        tilt = math.degrees(math.acos(float(np.clip(z_axis[2], -1, 1))))
        wheels = np.array([transforms[bidx[f"{w}_wheel_drive_link"], :3] for w in ("left","right","rear")])
        margins = []
        for i in range(3):
            a,b = wheels[i,:2], wheels[(i+1)%3,:2]
            e=b-a
            n=np.array([-e[1],e[0]])/np.linalg.norm(e)
            sign=np.sign(np.dot(wheels[:,:2].mean(0)-a,n))
            margins.append(float(np.dot(com[:2]-a,n)*sign))
        joint_positions=np.asarray(robot.get_joint_positions())[0]
        qx,qy,qz,qw=roots[3:]
        yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        joint_velocities=np.asarray(robot.get_joint_velocities())[0]
        return {"root_position_m": roots[:3].tolist(), "root_quaternion_xyzw":roots[3:].tolist(),
                "yaw_rad":yaw, "tilt_deg": tilt,
                "com_m": com.tolist(), "com_support_margin_m": min(margins),
                "wheel_centers_m": wheels.tolist(),
                "wheel_steer_rad":[float(joint_positions[idx[f"{w}_wheel_steer"]]) for w in ("left","right","rear")],
                "wheel_drive_rad_s":[float(joint_velocities[idx[f"{w}_wheel_drive"]]) for w in ("left","right","rear")],
                "gripper_rad": {s:[float(joint_positions[idx[f"gripper_{s}_joint{k}"]]) for k in range(1,5)] for s in ("l","r")}}

    max_tilt = 0.
    def step(seconds, world_velocity=None):
        global max_tilt
        for _ in range(round(seconds/DT)):
            if world_velocity is not None:
                measured=snapshot()
                angles,speeds=wheel_targets(*world_velocity, measured["yaw_rad"], current_steer=measured["wheel_steer_rad"])
                for w,angle,speed in zip(("left","right","rear"),angles,speeds):
                    pos[0,idx[f"{w}_wheel_steer"]]=angle
                    vel[0,idx[f"{w}_wheel_drive"]]=speed
            robot.set_joint_position_targets(pos)
            robot.set_joint_velocity_targets(vel)
            world.step(render=False)
            current = snapshot()
            max_tilt=max(max_tilt,current["tilt_deg"])
            if not np.isfinite(np.array(current["root_position_m"])).all():
                raise RuntimeError("non-finite root state")
    result["initial"]=snapshot()
    print("[SG2] settling", flush=True)
    step(2.)
    result["settle"]=snapshot()
    from collision_probe_helpers import probe_gripper_colliders
    result["gripper_collision_probe"]=probe_gripper_colliders(stage, "/World/Robot", robot)
    # Both hands: interior command and common URDF boundary.
    result["gripper_tests"]=[]
    for command in (.8, 1.0, 0.):
        for s in ("l","r"):
            pos[0,idx[f"gripper_{s}_joint1"]]=command
        step(1.5)
        snap=snapshot()
        fingers=snap["gripper_rad"]
        result["gripper_tests"].append({"command_rad":command,"positions_rad":fingers,
          "max_follow_error_rad":max(abs(values[k]-values[0]) for values in fingers.values() for k in range(1,4)),
          "max_master_error_rad":max(abs(values[0]-command) for values in fingers.values())})
    start_state=snapshot()
    start=np.asarray(start_state["root_position_m"])
    print("[SG2] forward drive", flush=True)
    for w in ("left","right","rear"):
        vel[0,idx[f"{w}_wheel_drive"]]=SPEED
    step(3., world_velocity=(RADIUS*SPEED,0.) if args.control == "world" else None)
    finish=np.asarray(snapshot()["root_position_m"])
    result["forward"]={"delta_m":(finish-start).tolist(),"expected_distance_m":RADIUS*SPEED*3.,"start_state":start_state,"state":snapshot()}
    for w in ("left","right","rear"):
        vel[0,idx[f"{w}_wheel_drive"]]=0
        pos[0,idx[f"{w}_wheel_steer"]]=math.pi/2
    step(1.)
    start_state=snapshot()
    start=np.asarray(start_state["root_position_m"])
    print("[SG2] lateral drive", flush=True)
    for w in ("left","right","rear"):
        vel[0,idx[f"{w}_wheel_drive"]]=SPEED
    step(2., world_velocity=(0.,RADIUS*SPEED) if args.control == "world" else None)
    finish=np.asarray(snapshot()["root_position_m"])
    result["lateral"]={"delta_m":(finish-start).tolist(),"expected_distance_m":RADIUS*SPEED*2.,"start_state":start_state,"state":snapshot()}
    result["max_tilt_deg"]=max_tilt
    if trial:
        forward=np.asarray(result["forward"]["delta_m"])
        lateral=np.asarray(result["lateral"]["delta_m"])
        result["checks"]={
          "settle_tilt_below_5deg":result["settle"]["tilt_deg"]<5,
          "com_inside_support":result["settle"]["com_support_margin_m"]>0,
          "wheel_centers_near_radius":max(abs(w[2]-RADIUS) for w in result["settle"]["wheel_centers_m"])<.01,
          "forward_within_15percent":abs(abs(forward[0])-RADIUS*SPEED*3.)<.15*RADIUS*SPEED*3.,
          "forward_cross_axis_below_0.1m":abs(forward[1])<.1,
          "lateral_within_15percent":abs(abs(lateral[1])-RADIUS*SPEED*2.)<.15*RADIUS*SPEED*2.,
          "lateral_cross_axis_below_0.1m":abs(lateral[0])<.1,
          "gripper_follow_below_0.02rad":max(t["max_follow_error_rad"] for t in result["gripper_tests"])<.02,
          "gripper_master_below_0.03rad":max(t["max_master_error_rad"] for t in result["gripper_tests"])<.03,
          "all_eight_finger_colliders_present":result["gripper_collision_probe"].get("all_eight_finger_colliders_confirmed",False),
          "both_gripper_base_colliders_present":result["gripper_collision_probe"].get("both_gripper_base_colliders_confirmed",False),
        }
    else:
        result["checks"]={"world_anchor_blocks_translation":max(np.linalg.norm(result[k]["delta_m"]) for k in ("forward","lateral"))<1e-4}
    result["checks"]={key:bool(value) for key,value in result["checks"].items()}
    result["passed"]=all(result["checks"].values())
except Exception:
    result["passed"]=False
    result["exception"]=traceback.format_exc()
finally:
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("RESULT_JSON: "+json.dumps(result,ensure_ascii=False),flush=True)
    app.close(exit_code=0 if result["passed"] else 1)

raise SystemExit(0 if result["passed"] else 1)
