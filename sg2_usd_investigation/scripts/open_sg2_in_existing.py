"""Open SG2 controls INSIDE an existing Isaac Sim 6.0.1 Script Editor.

After File > New, run:
    import runpy
    runpy.run_path("/home/robotis/workspaces/hx5_isaac/cyclo_lab/sg2_usd_investigation/scripts/open_sg2_in_existing.py")

No SimulationApp is created, and the existing application is never closed.
The tested standalone scene/UI code is adapted to the asynchronous Kit lifecycle.
Nonempty scenes are rejected rather than replaced. Official assets are referenced only.
"""
import asyncio
import builtins
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import omni.kit.app
import omni.timeline

folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(folder / "runtime"))
from swerve_controller import wheel_targets
app = omni.kit.app.get_app()
args = SimpleNamespace(demo_once=bool(globals().get("SG2_DEMO_ONCE", False)),
                       exit_after_demo=bool(globals().get("SG2_EXIT_AFTER_DEMO", False)))
import numpy as np
import omni.ui as ui
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import create_viewport_window, capture_viewport_to_file, get_active_viewport
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

DT = .005
output = folder / "results" / "manual_existing_wrist_601"
output.mkdir(parents=True, exist_ok=True)
asset = folder / "runtime" / "FFW_SG2_with_sensors.usda"
physics_asset = folder / "runtime" / "FFW_SG2_trial.usd"
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (asset, physics_asset)}
state = {"vx": 0., "vy": 0., "heading": 0., "speed": .35, "gripper": 0.,
         "follow": False, "demo": bool(args.demo_once), "demo_time": 0.,
         "demo_phase": None, "reset": False, "quit": False, "markers": True}
result = {"version": Path("/home/robotis/isaacsim/VERSION").read_text().strip(),
          "scope": "Script Editor async scene in existing Kit; not Isaac Lab or hardware validation",
          "robot_usd_sha256": before[str(physics_asset)], "events": [], "sensor_errors": {}}
exit_code = 0


def gains(name):
    if "wheel_" in name:
        return (10000, 100, 1000, 30) if name.endswith("steer") else (0, 100, 100, 30)
    if name == "lift_joint":
        return 10000, 100, 1000000, .2
    if name.startswith("head_"):
        return 150, 3, 30, 2
    if name.startswith("arm_"):
        n = int(name[-1])
        return (600, 30, 61.4, 15) if n <= 2 else ((600, 20, 31.7, 15) if n <= 6 else (200, 3, 5.1, 6))
    if name.startswith("gripper_"):
        return (100, 4, 30, 2.2) if name.endswith("1") else (0, 0, 20, 2.2)
    return None


def rotated(vector, quat_xyzw):
    u, w = quat_xyzw[:3], quat_xyzw[3]
    return vector + 2 * (w * np.cross(u, vector) + np.cross(u, np.cross(u, vector)))


def save_result():
    result["assets_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                                     for p, digest in before.items())
    (output / "session.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")



async def _open_sg2():
    try:
        context = omni.usd.get_context()
        current_stage = context.get_stage()
        if current_stage:
            occupied = [str(p.GetPath()) for p in current_stage.Traverse()
                        if (str(p.GetPath()).startswith("/World/")
                            or p.HasAuthoredReferences()
                            or (p.IsA(UsdGeom.Gprim) and not p.IsA(UsdGeom.Camera)))
                        and not str(p.GetPath()).startswith(("/Render", "/Replicator", "/OmniverseKit_"))]
            if occupied:
                raise RuntimeError("SG2: first use File > New for an empty scene; save your existing work. This launcher did not replace the current scene.")
        omni.timeline.get_timeline_interface().stop()
        await app.next_update_async()
        World.clear_instance()
        await context.new_stage_async()
        # This pinned 6.0.1 legacy constructor otherwise calls app.update()
        # synchronously inside our coroutine. Defer initialization to its public
        # async API; restore the flag before yielding or running other tasks.
        launch_flag = builtins.ISAAC_LAUNCHED_FROM_TERMINAL
        try:
            builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True
            world = World(physics_dt=DT, rendering_dt=.015, stage_units_in_meters=1, backend="numpy", device="cpu")
        finally:
            builtins.ISAAC_LAUNCHED_FROM_TERMINAL = launch_flag
        await world.initialize_simulation_context_async()
        world.scene.add_default_ground_plane(static_friction=1., dynamic_friction=1.)
        add_reference_to_stage(str(asset), "/World/Robot")
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robot")).AddTranslateOp(opSuffix="manual_spawn").Set(Gf.Vec3d(0, 0, .02))
        roots = []
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith("/World/Robot"):
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                articulation.CreateEnabledSelfCollisionsAttr(True)
                articulation.CreateSolverPositionIterationCountAttr(32)
                articulation.CreateSolverVelocityIterationCountAttr(1)
                roots.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                parameters = gains(prim.GetName())
                if parameters is None:
                    continue
                kp, kd, force, vmax = parameters
                angular = prim.IsA(UsdPhysics.RevoluteJoint)
                conversion = math.pi / 180 if angular else 1
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular" if angular else "linear")
                drive.CreateStiffnessAttr(kp * conversion)
                drive.CreateDampingAttr(kd * conversion)
                drive.CreateMaxForceAttr(force)
                drive.CreateTypeAttr("force")
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(vmax / conversion)
        if len(roots) != 1:
            raise RuntimeError(f"Expected one articulation root: {roots}")

        light = UsdLux.DomeLight.Define(stage, "/World/ManualLight")
        light.CreateIntensityAttr(1200.)
        # Fixed obstacles give the actual mounted camera and LiDAR something to see.
        for name, xyz, scale, color in (
            ("RedTarget", (4., 0., .8), (.8, 1.2, 1.6), (.8, .08, .05)),
            ("BlueTarget", (1., -3., .6), (2., .3, 1.2), (.05, .3, .8)),
            ("RearTarget", (-3., 1., .6), (.3, 2., 1.2), (.5, .5, .5)),
        ):
            cube = UsdGeom.Cube.Define(stage, "/World/" + name)
            cube.CreateSizeAttr(1.)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
            xform.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        com_marker = UsdGeom.Sphere.Define(stage, "/World/MeasuredCoM")
        com_marker.CreateRadiusAttr(.025)
        com_marker.CreateDisplayColorAttr([Gf.Vec3f(1, .05, .05)])
        marker_position = UsdGeom.Xformable(com_marker).AddTranslateOp()
        robot = world.scene.add(Articulation(roots[0], name="manual_sg2"))
        await world.reset_async()
        indices = {name: i for i, name in enumerate(robot.dof_names)}
        body_indices = {name: i for i, name in enumerate(robot.body_names)}
        targets = np.zeros((1, len(indices)), dtype=np.float32)
        speeds = np.zeros_like(targets)
        robot.set_joint_positions(targets)
        robot.set_joint_velocities(speeds)
        masses = np.asarray(robot.get_body_masses())[0]
        local_com = np.asarray(robot.get_body_coms()[0])[0]

        def snapshot():
            root = np.asarray(robot._physics_view.get_root_transforms())[0]
            links = np.asarray(robot._physics_view.get_link_transforms())[0]
            q = np.asarray(robot.get_joint_positions())[0]
            centers = np.array([link[:3] + rotated(c, link[3:]) for link, c in zip(links, local_com)])
            com = (centers * masses[:, None]).sum(0) / masses.sum()
            qx, qy, qz, qw = root[3:]
            yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
            tilt = math.degrees(math.acos(float(np.clip(rotated(np.array([0., 0., 1.]), root[3:])[2], -1, 1))))
            return {"root_position_m": root[:3].tolist(), "com_m": com.tolist(), "yaw_rad": yaw, "tilt_deg": tilt,
                    "steer_rad": [float(q[indices[f"{w}_wheel_steer"]]) for w in ("left", "right", "rear")],
                    "gripper_rad": {s: [float(q[indices[f"gripper_{s}_joint{k}"]]) for k in range(1, 5)] for s in ("l", "r")}}

        def stop():
            state.update(vx=0., vy=0., demo=False)

        def command(x, y):
            state.update(vx=x * state["speed"], vy=y * state["speed"], heading=snapshot()["yaw_rad"], demo=False)

        def home_view(close_up=False):
            viewport = main_viewport
            if viewport:
                viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
            if close_up:
                links = np.asarray(robot._physics_view.get_link_transforms())[0]
                index = body_indices["gripper_l_rh_p12_rn_base"]
                target = links[index, :3]
                eye = target + np.array([.55, -.55, .35])
            else:
                root = np.array(snapshot()["root_position_m"])
                target, eye = root + [0, 0, .65], root + [2.7, -3.2, 2.2]
            set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp", viewport_api=main_viewport)

        camera_path = ("/World/Robot/head_link2/zed_camera_link/zed_camera_center/"
                       "zed_left_camera_frame/zed_left_camera_optical_frame/TrialRgbDepthCamera")
        main_viewport = get_active_viewport()
        home_view()
        sensor_view = create_viewport_window("SG2 Head RGB (trial calibration)", width=480, height=320,
                                             camera_path=Sdf.Path(camera_path), position_x=1100, position_y=80)
        wrist_views = {}
        for side, y in (("left", 425), ("right", 685)):
            path = f"/World/Robot/camera_{side}_link/WristDepthOpticalFrame/TrialWristRgbDepthCamera"
            if not stage.GetPrimAtPath(path):
                raise ValueError(f"Missing wrist camera; rerun sensors_wrist validation: {path}")
            wrist_views[side] = create_viewport_window(
                f"SG2 {side.title()} Wrist RGB (nominal D405)", width=360, height=240,
                camera_path=Sdf.Path(path), position_x=1100, position_y=y)
        result["wrist_camera_paths"] = {side: str(view.viewport_api.camera_path)
                                        for side, view in wrist_views.items()}

        def toggle_wrist(side):
            wrist_views[side].visible = not wrist_views[side].visible

        # Preserve the authored sensor transforms. These wrappers only attach outputs.
        enable_extension("isaacsim.sensors.rtx.nodes")
        from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
        lidar_runtimes, lidar_annotators = {}, {}
        for side in ("l", "r"):
            path = f"/World/Robot/lidar_{side}_link/TrialGenericLidar"
            author = Lidar(path, reset_xform_op_properties=False, accumulate_outputs=None, aux_output_level="BASIC")
            sensor = LidarSensor(author, annotators=[])
            lidar_annotators[side] = sensor.attach_annotators("generic-model-output")["generic-model-output"]
            sensor.attach_writer("draw-point-cloud", size=.015, color=[0., 1., .4, 1.], doTransform=False)
            lidar_runtimes[side] = sensor

        controls = ui.Window("SG2 Manual Control - separate trial", width=365, height=535, position_x=20, position_y=80)
        with controls.frame:
            with ui.VStack(spacing=6):
                ui.Label("WORLD axes: X forward / Y left", height=22)
                ui.Label("Click motion; click STOP to stop.", height=20)
                ui.Label("Speed (m/s); applies to next motion", height=20)
                speed_slider = ui.FloatSlider(min=.05, max=.65, step=.05, height=24)
                speed_slider.model.set_value(state["speed"])
                speed_slider.model.add_value_changed_fn(lambda model: state.update(speed=model.get_value_as_float()))
                with ui.HStack(height=32):
                    ui.Button("+X Forward", clicked_fn=lambda: command(1, 0))
                    ui.Button("-X Back", clicked_fn=lambda: command(-1, 0))
                with ui.HStack(height=32):
                    ui.Button("+Y Left", clicked_fn=lambda: command(0, 1))
                    ui.Button("-Y Right", clicked_fn=lambda: command(0, -1))
                ui.Button("STOP", clicked_fn=stop, height=36)
                ui.Label("Both grippers: 0 to 1 rad", height=20)
                grip_slider = ui.FloatSlider(min=0., max=1., step=.05, height=24)
                grip_slider.model.add_value_changed_fn(lambda model: state.update(gripper=model.get_value_as_float()))
                with ui.HStack(height=28):
                    ui.Button("Grip 0", clicked_fn=lambda: grip_slider.model.set_value(0.))
                    ui.Button("Grip 0.8", clicked_fn=lambda: grip_slider.model.set_value(.8))
                with ui.HStack(height=28):
                    ui.Button("Home view", clicked_fn=home_view)
                    ui.Button("Hand close-up", clicked_fn=lambda: home_view(True))
                with ui.HStack(height=28):
                    ui.Button("Left wrist RGB", clicked_fn=lambda: toggle_wrist("left"))
                    ui.Button("Right wrist RGB", clicked_fn=lambda: toggle_wrist("right"))
                with ui.HStack(height=28):
                    ui.Button("Follow ON/OFF", clicked_fn=lambda: state.update(follow=not state["follow"]))
                    ui.Button("CoM dot ON/OFF", clicked_fn=lambda: state.update(markers=not state["markers"]))
                with ui.HStack(height=28):
                    ui.Button("Demo once", clicked_fn=lambda: state.update(demo=True, demo_time=0., demo_phase=None))
                    ui.Button("Reset robot", clicked_fn=lambda: state.update(reset=True))
                status_label = ui.Label("Settling...", height=60, word_wrap=True)
                ui.Label("Red dot = measured CoM; green = RTX LiDAR", height=20)
                ui.Label("Temporary gains/profile; original files untouched", height=20)

        # Settle using exactly the physics timestep used by the regression harness.
        for _ in range(134):
            robot.set_joint_position_targets(targets)
            robot.set_joint_velocity_targets(speeds)
            # render=True runs the configured three 5-ms substeps internally.
            await app.next_update_async()
        result["settled"] = snapshot()
        result["status"] = "ready"
        save_result()
        print("SG2_GUI_READY: manual buttons active; original assets unchanged", flush=True)
        frame = 0
        captures = []
        previous_phase = None
        while controls.visible and not state["quit"] and omni.usd.get_context().get_stage() == stage:
            started = time.monotonic()
            if state["reset"]:
                stop()
                await world.reset_async()
                targets.fill(0.)
                speeds.fill(0.)
                robot.set_joint_positions(targets)
                robot.set_joint_velocities(speeds)
                grip_slider.model.set_value(0.)
                state["reset"] = False
                home_view()
            if not world.is_playing():
                stop()
                await app.next_update_async()
                continue
            current = snapshot()
            if state["demo"]:
                t = state["demo_time"]
                if t < 2.5:
                    phase, vx, vy, grip = "forward", .35, 0., 0.
                elif t < 3.5:
                    phase, vx, vy, grip = "stop", 0., 0., 0.
                elif t < 6.5:
                    phase, vx, vy, grip = "lateral", 0., .35, 0.
                elif t < 8.5:
                    phase, vx, vy, grip = "grip_0.8", 0., 0., .8
                elif t < 10.5:
                    phase, vx, vy, grip = "grip_0", 0., 0., 0.
                else:
                    phase, vx, vy, grip = "complete", 0., 0., 0.
                    state["demo"] = False
                if phase != state["demo_phase"]:
                    result["events"].append({"phase": phase, "snapshot": current})
                    state.update(demo_phase=phase, heading=current["yaw_rad"])
                    print("SG2_GUI_DEMO: " + phase, flush=True)
                    if phase == "complete":
                        result["status"] = "demo_complete_controls_ready"
                        result["demo_final"] = current
                        save_result()
                        captures.append(capture_viewport_to_file(main_viewport, str(output / "robot_view.png")))
                        captures.append(capture_viewport_to_file(sensor_view.viewport_api, str(output / "head_rgb.png")))
                        for side, view in wrist_views.items():
                            captures.append(capture_viewport_to_file(view.viewport_api, str(output / f"wrist_{side}_rgb.png")))
                        previous_phase = "complete"
                state.update(vx=vx, vy=vy, gripper=grip)
                grip_slider.model.set_value(grip)
            for side in ("l", "r"):
                targets[0, indices[f"gripper_{side}_joint1"]] = state["gripper"]
            if abs(state["vx"]) + abs(state["vy"]) > 0:
                angles, velocities = wheel_targets(state["vx"], state["vy"], current["yaw_rad"],
                                                   heading_rad=state["heading"], current_steer=current["steer_rad"])
                aligned = max(abs(a-b) for a, b in zip(angles, current["steer_rad"])) < .18
                for wheel, angle, speed in zip(("left", "right", "rear"), angles, velocities):
                    targets[0, indices[f"{wheel}_wheel_steer"]] = angle
                    speeds[0, indices[f"{wheel}_wheel_drive"]] = speed if aligned else 0.
            else:
                for wheel in ("left", "right", "rear"):
                    speeds[0, indices[f"{wheel}_wheel_drive"]] = 0.
            simulated_before = world.current_time
            robot.set_joint_position_targets(targets)
            robot.set_joint_velocity_targets(speeds)
            await app.next_update_async()
            if state["demo"]:
                state["demo_time"] += max(0., world.current_time - simulated_before)
            if frame % 10 == 0:
                current = snapshot()
                marker_position.Set(Gf.Vec3d(*current["com_m"]))
                com_marker.GetVisibilityAttr().Set("inherited" if state["markers"] else "invisible")
                if state["follow"]:
                    home_view()
                counts = {}
                for side, annotator in lidar_annotators.items():
                    try:
                        raw = annotator.get_data(device="cpu")
                        raw = raw.get("data") if isinstance(raw, dict) else raw
                        if raw is not None and getattr(raw, "size", 0):
                            counts[side] = int(parse_generic_model_output_data(raw).numElements)
                    except Exception as exc:
                        result["sensor_errors"][side] = str(exc)
                if counts:
                    result["lidar_elements"] = counts
                xyz = current["root_position_m"]
                status_label.text = (f"Base: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) m\n"
                                     f"Tilt: {current['tilt_deg']:.2f} deg | LiDAR: {counts}\n"
                                     f"Grip L/R: {current['gripper_rad']['l'][0]:.3f} / {current['gripper_rad']['r'][0]:.3f} rad")
                if not np.isfinite(xyz).all():
                    raise RuntimeError("Non-finite robot state")
            if frame % 200 == 0:
                result["latest"] = snapshot()
                save_result()
            frame += 1
            if args.exit_after_demo and previous_phase == "complete" and frame > 740:
                break
        stop()
        if omni.usd.get_context().get_stage() == stage and world.is_playing():
            result["latest"] = snapshot()
            speeds.fill(0.)
            robot.set_joint_velocity_targets(speeds)
            omni.timeline.get_timeline_interface().pause()
        for view in wrist_views.values():
            view.visible = False
        sensor_view.visible = False
        controls.visible = False
        result["status_at_close"] = result["status"]
    except Exception:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        save_result()

_previous_window = ui.Workspace.get_window("SG2 Manual Control - separate trial")
if _previous_window and _previous_window.visible:
    print("SG2 Manual Control is already open; use its buttons.")
    _sg2_task = None
else:
    _sg2_task = asyncio.ensure_future(_open_sg2())
