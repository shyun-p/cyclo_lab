#!/usr/bin/env python3
"""Attach trial RTX sensors to SG2 mounts and measure known test targets.

Run with Isaac Sim 6.0.1's python.sh, never a system Python importing Kit:
    /home/robotis/isaacsim/python.sh sensor_probe_601.py \
        --usd /absolute/path/to/FFW_SG2_trial.usd --out-dir /absolute/output

This creates a separate sensor fixture. It never saves the referenced robot.
Sensor intrinsics, LiDAR scan profile, and noise are GENERIC TRIAL VALUES.
The test proves sensor authoring/rendering/data retrieval, not hardware fidelity.
The robot is frozen explicitly for this test; wheel/gripper physics must be
validated separately with normal gravity and enabled rigid bodies.

API sources: the installed 6.0.1 standalone examples create_camera_basic.py,
inspect_lidar_gmo.py, and sensors.experimental.rtx.impl.{lidar,_sensor_base}.
The proposed sensor code and the difference from the original are commented
at the bottom of this new file for review, as requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=180)
    args, _ = parser.parse_known_args()
    args.usd = args.usd.resolve(strict=True)
    args.out_dir = args.out_dir.resolve()
    if not 60 <= args.frames <= 1200:
        parser.error("--frames must be between 60 and 1200")
    return args


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_dynamic_sensor_asset(fixture_stage, candidate_path, sensor_paths):
    """Export only sensor prims over the candidate's unchanged dynamics.

    The fixture's gravity/body/joint overrides, targets, lights, render products,
    and annotator graphs are deliberately absent from this new asset's layer.
    This function does not claim that combined dynamics + sensors were tested.
    """
    from pxr import Sdf, Usd, UsdGeom

    candidate_path = Path(candidate_path)
    output_path = candidate_path.parent / "FFW_SG2_with_sensors.usda"
    review_path = candidate_path.parent / "sensors.review.usda"
    clean_stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(clean_stage, 1.0)
    UsdGeom.SetStageUpAxis(clean_stage, UsdGeom.Tokens.z)
    clean_root = UsdGeom.Xform.Define(clean_stage, "/ffw_sg2_follower").GetPrim()
    clean_stage.SetDefaultPrim(clean_root)
    clean_root.GetReferences().AddReference("./" + candidate_path.name)
    clean_layer = clean_stage.GetRootLayer()
    base_text = clean_layer.ExportToString()
    copied_paths = []
    fixture_prefix = Sdf.Path("/World/SG2")
    clean_prefix = Sdf.Path("/ffw_sg2_follower")
    for path in sensor_paths:
        source_path = Sdf.Path(path)
        if not source_path.HasPrefix(fixture_prefix):
            raise ValueError(f"Sensor path is outside candidate root: {source_path}")
        destination = source_path.ReplacePrefix(fixture_prefix, clean_prefix)
        clean_stage.OverridePrim(destination.GetParentPath())
        # Ensure every ancestor exists in the authored Sdf layer as well as
        # in USD composition before CopySpec creates the sensor's child spec.
        Sdf.CreatePrimInLayer(clean_layer, destination.GetParentPath())
        if not Sdf.CopySpec(fixture_stage.GetRootLayer(), source_path, clean_layer, destination):
            raise RuntimeError(f"Cannot copy sensor prim: {source_path}")
        copied_paths.append(str(destination))
    if len(copied_paths) != 6:
        raise ValueError(f"Expected four cameras and two lidar subtrees, got {copied_paths}")
    # Check authored layer content, not composition: inherited robot physics is
    # expected, while fixture-specific physics overrides must never leak in.
    authored_text = clean_layer.ExportToString()
    forbidden = (
        "gravityMagnitude", "physics:rigidBodyEnabled", "physxRigidBody:disableGravity",
        "physics:jointEnabled", "physxArticulation:articulationEnabled",
        "PhysicsScene", "RenderProduct", "CameraTarget", "LidarTarget", "TrialLight",
    )
    leaked = [name for name in forbidden if name in authored_text]
    if leaked:
        raise ValueError(f"Fixture overrides leaked into dynamic sensor asset: {leaked}")
    clean_layer.Save()
    review_text = (
        base_text
        + "\n# ---------------------------------------------------------------------------\n"
        + "# 기존: trial USD의 카메라/LiDAR 장착 좌표만 참조; 실제 센서 prim 없음.\n"
        + "# 아래는 검토용 주석 처리 제안. 공식 파일 및 위 기존 참조는 수정하지 않음.\n"
        + "# 제안: 머리 Camera 2개 + 손목 Camera 2개 + OmniLidar 2개를 공식 장착 좌표에 추가.\n"
        + "# 값은 일반 시험용이며 실제 카메라/LiDAR 보정 및 제품 사양은 확인 필요.\n"
        + "# gravity/body/joint 설정은 trial USD에서 그대로 상속. fixture 설정은 제외.\n"
        + "# 이 파일은 제안 코드가 모두 주석이며 실행본은 FFW_SG2_with_sensors.usda.\n"
        + "# 센서 거리 fixture와 동적 GUI의 검증 범위는 별도 결과 파일을 확인.\n"
        + "# ---------------------------------------------------------------------------\n"
        + "\n".join("# " + line for line in authored_text.splitlines()) + "\n"
    )
    review_path.write_text(review_text, encoding="utf-8")
    return {
        "robot_with_sensors_usda": str(output_path),
        "sensor_review_usda": str(review_path),
        "reference": "./" + candidate_path.name,
        "copied_sensor_paths": copied_paths,
        "fixture_overrides_excluded": list(forbidden),
        "physics_scope": "inherits candidate dynamics; combined dynamics + sensor simulation not verified",
    }


def run_probe(app, args, result):
    # Import Kit-dependent modules only after SimulationApp starts.
    import numpy as np
    import omni.timeline
    import omni.usd
    from PIL import Image
    from isaacsim.sensors.experimental.rtx import (
        CameraSensor,
        Lidar,
        LidarSensor,
        RtxCamera,
        parse_generic_model_output_data,
    )
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    source = Usd.Stage.Open(str(args.usd))
    if not source or not source.GetDefaultPrim():
        raise ValueError("Candidate USD must have a valid defaultPrim")
    if abs(UsdGeom.GetStageMetersPerUnit(source) - 1.0) > 1e-6:
        raise ValueError("Probe expects SG2 authored in meters")
    root = UsdGeom.Xform.Define(stage, "/World/SG2").GetPrim()
    root.GetReferences().AddReference(str(args.usd))

    # Rendering fixture overrides are authored on this new stage only.
    # Freezing physics prevents changing sensor-target distances while probing.
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(0.0)
    frozen_bodies, frozen_joints, disabled_roots = [], [], []
    for prim in list(Usd.PrimRange(root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
            frozen_bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            frozen_joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateArticulationEnabledAttr(False)
            disabled_roots.append(str(prim.GetPath()))
    result["fixture_physics"] = {
        "purpose": "static sensor-distance fixture; separate from dynamics validation",
        "gravity_m_s2": 0.0,
        "rigid_body_enabled": False,
        "rigid_body_disable_gravity": True,
        "joint_enabled": False,
        "articulation_enabled": False,
        "body_paths": frozen_bodies,
        "joint_paths": frozen_joints,
        "root_paths": disabled_roots,
    }

    light = UsdLux.DomeLight.Define(stage, "/World/TrialLight")
    light.CreateIntensityAttr(1800.0)

    def transform_for(prim):
        return UsdGeom.XformCache().GetLocalToWorldTransform(prim)

    def target_box(path, frame, center, dimensions, color):
        # The target is defined in a sensor frame and then placed in the world.
        # Thus the test still works when a mount has a non-identity rotation.
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        local = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*center))
        xform = UsdGeom.Xformable(cube)
        xform.AddTransformOp().Set(local * frame)
        xform.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return str(cube.GetPath())

    def annotate_sensor(prim, description):
        prim.SetCustomDataByKey("sg2_investigation", {
            "hardware_calibration": "unknown; generic trial only",
            "description": description,
        })

    # Real optical mounts present in the regenerated v4 SG2 asset.
    cameras, camera_records = {}, {}
    for side in ("left", "right"):
        mount_path = (
            "/World/SG2/head_link2/zed_camera_link/zed_camera_center/"
            f"zed_{side}_camera_frame/zed_{side}_camera_optical_frame"
        )
        mount = stage.GetPrimAtPath(mount_path)
        if not mount:
            raise ValueError(f"Missing optical mount: {mount_path}")
        camera_path = mount_path + "/TrialRgbDepthCamera"
        authoring = RtxCamera(camera_path, tick_rate=30.0)
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        # ROS optical +Z forward/+Y down -> USD camera -Z forward/+Y up.
        local_xform = UsdGeom.Xformable(camera)
        local_xform.ClearXformOpOrder()
        local_xform.AddRotateXOp().Set(180.0)
        # Set USD optical fields directly to keep their unit convention explicit.
        # FOV depends on the aperture/focal ratio. These are generic trial values.
        camera.CreateFocalLengthAttr(12.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.71625)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
        camera.CreateFocusDistanceAttr(3.0)
        annotate_sensor(camera.GetPrim(), "head RGB + ideal image-plane depth")
        sensor = CameraSensor(
            authoring,
            resolution=(240, 320),
            annotators=["rgb", "distance_to_image_plane"],
        )
        # CameraSensor accesses RtxCamera.camera, whose first Camera wrapper
        # construction resets the xform stack in this installed 6.0.1 build.
        # Set the local optical conversion AFTER all wrappers are initialized.
        # No mount translation or original URDF orientation is changed.
        local_xform.ClearXformOpOrder()
        local_xform.AddRotateXOp().Set(180.0)
        camera_world = transform_for(camera.GetPrim())
        mount_world = transform_for(mount)
        world_forward = camera_world.TransformDir(Gf.Vec3d(0, 0, -1))
        optical_forward = mount_world.TransformDir(Gf.Vec3d(0, 0, 1))
        alignment = Gf.Dot(world_forward.GetNormalized(), optical_forward.GetNormalized())
        if alignment < 0.99999:
            raise ValueError(f"Camera optical-axis conversion failed at {camera_path}: {alignment}")
        cameras[side] = sensor
        camera_records[side] = {
            "mount_path": mount_path,
            "sensor_path": camera_path,
            "prim_type": str(camera.GetPrim().GetTypeName()),
            "resolution_hw": [240, 320],
            "focal_length_usd": 12.0,
            "horizontal_aperture_usd": 20.955,
            "vertical_aperture_usd": 15.71625,
            "local_optical_to_usd_rotation_x_deg": 180.0,
            "orientation_authoring_order": "local Rx180 after CameraSensor initializes/reset its Camera wrapper",
            "world_forward": list(world_forward),
            "mount_optical_world_forward": list(optical_forward),
            "optical_forward_alignment_dot": float(alignment),
            "world_position_m": list(transform_for(camera.GetPrim()).ExtractTranslation()),
            "expected_center_image_plane_depth_m": 2.6,
            "depth_tolerance_m": 0.10,
        }

    # One red box covers both stereo center rays. Its front face is 2.6 m
    # from the left camera, with a wall at 5 m giving background depth.
    left_frame = transform_for(stage.GetPrimAtPath(camera_records["left"]["mount_path"]))
    camera_target = target_box(
        "/World/CameraTarget", left_frame, (0, 0, 3.0), (0.8, 0.8, 0.8), (0.9, 0.05, 0.05)
    )
    camera_background = target_box(
        "/World/CameraBackground", left_frame, (0, 0, 5.0), (5.0, 4.0, 0.1), (0.2, 0.5, 0.8)
    )

    # Official ROBOTIS wrist mounts plus the manufacturer's nominal optical
    # axes. These two camera functions were absent from the earlier head probe.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
    from wrist_cameras import author_wrist_cameras, set_usd_optical_orientation
    wrist_targets = []
    for name, record in author_wrist_cameras(stage, "/World/SG2").items():
        author = record.pop("author")
        sensor = CameraSensor(author, resolution=tuple(record["resolution_hw"]),
                              annotators=["rgb", "distance_to_image_plane"])
        set_usd_optical_orientation(stage, record["sensor_path"])
        camera_frame = transform_for(stage.GetPrimAtPath(record["sensor_path"]))
        mount_frame = transform_for(stage.GetPrimAtPath(record["mount_path"]))
        forward = camera_frame.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
        body_forward = mount_frame.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
        record["optical_forward_alignment_dot"] = float(Gf.Dot(forward, body_forward))
        if record["optical_forward_alignment_dot"] < .99999:
            raise ValueError(f"Wrong D405 forward direction at {name}")
        record["world_position_m"] = list(camera_frame.ExtractTranslation())
        record["prim_type"] = "Camera"
        record["expected_center_image_plane_depth_m"] = .15
        record["depth_tolerance_m"] = .01
        cameras[name] = sensor
        camera_records[name] = record
        # Target+background use USD camera coordinates (-Zforward,+Yup).
        wrist_targets.append(target_box(
            f"/World/WristTarget_{name}", camera_frame, (0, 0, -.175),
            (.08, .06, .05), (.95, .35, .05)))
        wrist_targets.append(target_box(
            f"/World/WristBackground_{name}", camera_frame, (0, 0, -.405),
            (.6, .4, .01), (.05, .2, .65)))

    # A local one-channel ideal rotary profile creates a real OmniLidar and
    # avoids Lidar.create(config=...)'s remote Isaac asset lookup entirely.
    # The 0.1 m near range is a trial value, not ROBOTIS hardware calibration.
    lidar_attributes = {
        "omni:sensor:Core:scanType": "ROTARY",
        "omni:sensor:Core:rayType": "IDEALIZED",
        "omni:sensor:Core:scanRateBaseHz": 10,
        "omni:sensor:Core:patternFiringRateHz": 3600,
        "omni:sensor:Core:numberOfEmitters": 1,
        "omni:sensor:Core:numberOfChannels": 1,
        "omni:sensor:Core:maxReturns": 1,
        "omni:sensor:Core:nearRangeM": 0.1,
        "omni:sensor:Core:farRangeM": 20.0,
        "omni:sensor:Core:rangeAccuracyM": 0.0,
        "omni:sensor:Core:azimuthErrorStd": 0.0,
        "omni:sensor:Core:elevationErrorStd": 0.0,
        "omni:sensor:Core:elementsCoordsType": "CARTESIAN",
        "omni:sensor:Core:outputFrameOfReference": "WORLD",
        "omni:sensor:Core:emitterState:s001:azimuthDeg": [0.0],
        "omni:sensor:Core:emitterState:s001:elevationDeg": [0.0],
        "omni:sensor:Core:emitterState:s001:fireTimeNs": [0],
        # Schema's emitter channel convention starts at 1 (default is 1..128).
        "omni:sensor:Core:emitterState:s001:channelId": [1],
    }
    lidars, lidar_records, lidar_frames, lidar_annotators = {}, {}, {}, {}
    for side in ("l", "r"):
        mount_path = f"/World/SG2/lidar_{side}_link"
        mount = stage.GetPrimAtPath(mount_path)
        if not mount:
            raise ValueError(f"Missing LiDAR mount: {mount_path}")
        lidar_path = mount_path + "/TrialGenericLidar"
        authoring = Lidar(
            lidar_path,
            accumulate_outputs=True,
            aux_output_level="BASIC",
            tick_rate=10.0,
            attributes=lidar_attributes,
        )
        lidar_prim = stage.GetPrimAtPath(lidar_path)
        # Author identity locally; an API world-orientation setter would erase
        # the mount's rotation and make the attachment incorrect when moving.
        lidar_xform = UsdGeom.Xformable(lidar_prim)
        lidar_xform.ClearXformOpOrder()
        lidar_xform.AddTranslateOp().Set(Gf.Vec3d(0.0))
        annotate_sensor(lidar_prim, "local GENERIC trial 2D rotary LiDAR")
        sensor = LidarSensor(authoring, annotators=[])
        # attach_annotators returns the public Replicator annotator object.
        # GenericModelOutput is registered/attached on CPU by this installed
        # API, but LidarSensor.get_data unconditionally requests CUDA. Fetch
        # explicitly on CPU to keep buffer extraction on its attached device.
        lidar_annotators[side] = sensor.attach_annotators("generic-model-output")["generic-model-output"]
        lidars[side] = sensor
        frame = transform_for(lidar_prim)
        lidar_frames[side] = frame
        target = target_box(
            f"/World/LidarTarget_{side}", frame, (4.0, 0, 0), (0.2, 2.0, 0.8), (0.8, 0.8, 0.8)
        )
        lidar_records[side] = {
            "mount_path": mount_path,
            "sensor_path": lidar_path,
            "prim_type": str(lidar_prim.GetTypeName()),
            "world_position_m": list(frame.ExtractTranslation()),
            "profile": "GENERIC TRIAL; ideal one-channel 360-degree 10-Hz rotary",
            "attributes": lidar_attributes,
            "retrieval_method": "public attach_annotators result; GenericModelOutput.get_data(device='cpu')",
            "target_path": target,
            "expected_wall_plane_local_x_m": 3.9,
            "wall_tolerance_m": 0.10,
        }

    result["camera"] = camera_records
    result["lidar"] = lidar_records
    result["camera_target_paths"] = [camera_target, camera_background, *wrist_targets]
    result["exported_sensor_assets"] = export_dynamic_sensor_asset(
        stage,
        args.usd,
        [record.get("copy_subtree_path", record["sensor_path"])
         for record in [*camera_records.values(), *lidar_records.values()]],
    )
    runtime_dir = args.out_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    scene_path = runtime_dir / "sensor_scene.usda"
    # Export authored additions/overrides with the candidate USD reference intact.
    # This is reviewable without creating a second flattened robot replacement.
    stage.GetRootLayer().Export(str(scene_path))
    result["scene_usda"] = str(scene_path)
    result["status"] = "sensors_authored"
    write_json(args.out_dir / "sensor_result.json", result)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_target_framerate(60.0)
    timeline.play()
    last_rgb, last_depth, last_points = {}, {}, {}
    read_errors = {}
    lidar_diagnostics = {
        side: {"empty_buffer_frames": 0, "nonempty_buffer_frames": 0,
               "maximum_buffer_size": 0, "maximum_gmo_elements": 0}
        for side in lidars
    }
    try:
        for frame_number in range(args.frames):
            app.update()
            for side, sensor in cameras.items():
                try:
                    rgb, _ = sensor.get_data("rgb")
                    depth, _ = sensor.get_data("distance_to_image_plane")
                    if rgb is not None and rgb.size:
                        last_rgb[side] = rgb.numpy().copy()
                    if depth is not None and depth.size:
                        last_depth[side] = depth.numpy().copy()
                except Exception as exc:
                    read_errors["camera_" + side] = f"{type(exc).__name__}: {exc}"
            for side, sensor in lidars.items():
                try:
                    raw = lidar_annotators[side].get_data(device="cpu")
                    if isinstance(raw, dict):
                        raw = raw.get("data")
                    buffer_size = int(getattr(raw, "size", 1)) if raw is not None else 0
                    if not buffer_size:
                        lidar_diagnostics[side]["empty_buffer_frames"] += 1
                        continue
                    lidar_diagnostics[side]["nonempty_buffer_frames"] += 1
                    lidar_diagnostics[side]["maximum_buffer_size"] = max(
                        buffer_size, lidar_diagnostics[side]["maximum_buffer_size"]
                    )
                    gmo = parse_generic_model_output_data(raw)
                    lidar_diagnostics[side]["maximum_gmo_elements"] = max(
                        int(gmo.numElements), lidar_diagnostics[side]["maximum_gmo_elements"]
                    )
                    if gmo.numElements:
                        points = np.column_stack((gmo.x, gmo.y, gmo.z)).copy()
                        if points.size:
                            last_points[side] = points
                except Exception as exc:
                    read_errors["lidar_" + side] = f"{type(exc).__name__}: {exc}"
            # Give both sensor rates one simulated second to produce data.
            # Keep the run bounded even when RTX is unavailable or returns no hits.
            if (frame_number >= 59 and len(last_rgb) == len(cameras)
                    and len(last_depth) == len(cameras) and len(last_points) == len(lidars)):
                break
    finally:
        timeline.stop()
    result["frames_run"] = frame_number + 1
    result["read_errors"] = read_errors
    result["lidar_buffer_diagnostics"] = lidar_diagnostics

    for side, record in camera_records.items():
        rgb, depth = last_rgb.get(side), last_depth.get(side)
        record["rgb_present"] = rgb is not None
        record["depth_present"] = depth is not None
        if rgb is not None:
            Image.fromarray(rgb.astype(np.uint8)).save(args.out_dir / f"camera_{side}_rgb.png")
            record["rgb_shape"] = list(rgb.shape)
            record["rgb_nonconstant"] = bool(np.ptp(rgb[..., :3]) > 0)
            record["rgb_path"] = str(args.out_dir / f"camera_{side}_rgb.png")
        if depth is not None:
            np.save(args.out_dir / f"camera_{side}_depth_m.npy", depth)
            image = np.squeeze(depth)
            positive = np.isfinite(image) & (image > 0.0)
            record["depth_shape"] = list(depth.shape)
            record["valid_depth_pixels"] = int(positive.sum())
            if positive.any():
                record["finite_depth_min_m"] = float(image[positive].min())
                record["finite_depth_max_m"] = float(image[positive].max())
            height, width = image.shape
            center = image[height // 2 - 2:height // 2 + 3, width // 2 - 2:width // 2 + 3]
            valid = center[np.isfinite(center) & (center > 0.0)]
            measured = float(np.median(valid)) if valid.size else None
            record["measured_center_depth_m"] = measured
            record["known_target_distance_pass"] = bool(
                measured is not None and abs(measured - record["expected_center_image_plane_depth_m"])
                <= record["depth_tolerance_m"]
            )
        record["pass"] = bool(
            record["rgb_present"] and record["depth_present"]
            and record.get("rgb_nonconstant") and record.get("known_target_distance_pass")
        )

    for side, record in lidar_records.items():
        points = last_points.get(side)
        record["data_present"] = points is not None
        if points is not None:
            finite = points[np.isfinite(points).all(axis=1)]
            frame = lidar_frames[side]
            inverse = frame.GetInverse()
            local = np.array(
                [list(inverse.Transform(Gf.Vec3d(*map(float, point)))) for point in finite], dtype=float
            ).reshape((-1, 3))
            ranges = np.linalg.norm(local, axis=1)
            valid = (ranges >= 0.1) & (ranges <= 20.1)
            local = local[valid]
            record["raw_point_count"] = int(len(points))
            record["valid_point_count"] = int(len(local))
            if len(local):
                record["range_min_m"] = float(np.linalg.norm(local, axis=1).min())
                record["range_max_m"] = float(np.linalg.norm(local, axis=1).max())
            # A valid result must include hits on the specific known target plane.
            # Nonempty self-hits or arbitrary background returns do not pass.
            wall = local[
                (np.abs(local[:, 0] - 3.9) <= record["wall_tolerance_m"])
                & (np.abs(local[:, 1]) <= 0.95)
                & (np.abs(local[:, 2]) <= 0.35)
            ]
            record["known_wall_hit_count"] = int(len(wall))
            record["known_target_distance_pass"] = bool(len(wall) >= 3)
            if len(wall):
                record["measured_wall_plane_local_x_m"] = float(np.median(wall[:, 0]))
            point_path = args.out_dir / f"lidar_{side}_world_points_m.npy"
            np.save(point_path, finite)
            record["points_path"] = str(point_path)
        record["pass"] = bool(record["data_present"] and record.get("known_target_distance_pass"))

    result["pass"] = all(r["pass"] for r in [*camera_records.values(), *lidar_records.values()])
    result["status"] = "passed" if result["pass"] else "failed"
    return 0 if result["pass"] else 2


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "starting",
        "candidate_usd": str(args.usd),
        "candidate_sha256_before": hashlib.sha256(args.usd.read_bytes()).hexdigest(),
        "requested_isaac_version": "6.0.1-rc.7 (installed runtime)",
        "calibration_status": "trial sensor models; ROBOTIS hardware parameters unknown",
        "validation_scope": "head stereo + both D405 wrist RGB/ideal depth; two OmniLidars; static fixture",
    }
    version = Path("/home/robotis/isaacsim/VERSION")
    if version.exists():
        result["installed_version"] = version.read_text().strip()
    result_path = args.out_dir / "sensor_result.json"
    write_json(result_path, result)
    from isaacsim import SimulationApp

    app = None
    try:
        app = SimulationApp({
            "headless": True,
            "renderer": "RaytracedLighting",
            "enable_motion_bvh": True,
            "multi_gpu": False,
            "width": 640,
            "height": 480,
        })
        exit_code = run_probe(app, args, result)
    except Exception as exc:
        result["status"] = "error"
        result["pass"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
        exit_code = 2
    finally:
        result["candidate_sha256_after"] = hashlib.sha256(args.usd.read_bytes()).hexdigest()
        result["candidate_unchanged"] = result["candidate_sha256_before"] == result["candidate_sha256_after"]
        write_json(result_path, result)
        print(json.dumps({"result": str(result_path), "status": result["status"], "pass": result.get("pass", False)}), flush=True)
        if app is not None:
            # The 6.0.1 fast-shutdown path exits inside close(). Pass the actual
            # verdict to preserve failure status instead of always returning 0.
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# 검토용 변경 제안 (공식 FFW_SG2.usd / FFW_SG2.py는 수정하지 않음)
# ---------------------------------------------------------------------------
# 기존: camera / lidar 이름의 Xform 또는 고정 링크만 있고 실제 센서 prim 없음.
# 제안: head optical frame 자식에 Camera + OmniSensorAPI를 생성하고
#       CameraSensor(..., annotators=["rgb", "distance_to_image_plane"]) 연결.
# 제안: lidar_l_link / lidar_r_link 자식에 OmniLidar를 생성하고
#       LidarSensor(..., annotators=["generic-model-output"]) 연결.
# 예시:
# camera = RtxCamera(".../zed_left_camera_optical_frame/TrialRgbDepthCamera")
# # ROS 광학좌표 +Z 전방을 USD 카메라 -Z 전방으로 변환: local Rx(180 deg).
# camera_sensor = CameraSensor(camera, resolution=(240, 320),
#                             annotators=["rgb", "distance_to_image_plane"])
# lidar = Lidar(".../lidar_l_link/TrialGenericLidar", attributes=lidar_attributes)
# lidar_sensor = LidarSensor(lidar, annotators=["generic-model-output"])
# 차이: 위치 좌표만 있는 상태에서 실제 렌더링과 데이터 취득이 가능한 센서로 변경.
# 한계: 카메라 내부 파라미터, 렌즈 왜곡, 실제 depth 오차/스테레오 방식,
#       LiDAR 제품명, 채널 수, 주사속도, 범위 및 노이즈는 실물 사양 확인 필요.
# 이 fixture의 gravity=0 / rigidBodyEnabled=false 설정은 센서 거리 검증에만 사용.
# 물리 테스트 및 실제 pick_place 환경에서는 동적 로봇과 정상 중력을 사용해야 함.
# 실행본 FFW_SG2_with_sensors.usda는 별도 trial USD를 상대 참조하고 센서 4개만 추가.
# sensors.review.usda 아래쪽에는 동일 추가 내용을 주석으로 보관하여 비교할 수 있음.
# 실행본에는 정지 fixture의 gravity/body/joint override 또는 표적/조명/RenderProduct 없음.
