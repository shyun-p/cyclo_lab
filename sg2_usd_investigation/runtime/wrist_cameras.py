"""Add RGB + ideal depth at ROBOTIS's existing D405 wrist mounts.

ROBOTIS mount: pinned SG2 URDF camera_left/right_link (arm_*_link7 child).
Nominal optical extrinsics: RealSense _d405.urdf.xacro depth/color optical joints.
https://raw.githubusercontent.com/realsenseai/realsense-ros/ros2-master/realsense2_description/urdf/_d405.urdf.xacro
Nominal FOV87x58 degrees: https://www.realsenseai.com/products/d405-series/
No calibrated distortion, stereo matching, noise, or physical mass is added.
Resolution/rate and renderer clipping are explicit simulation choices.
"""
import math


def author_wrist_cameras(stage, robot_path):
    from isaacsim.sensors.experimental.rtx import RtxCamera
    from pxr import Gf, UsdGeom

    records = {}
    for side in ("left", "right"):
        mount_path = f"{robot_path}/camera_{side}_link"
        mount = stage.GetPrimAtPath(mount_path)
        if not mount:
            raise ValueError(f"Missing official D405 wrist mount: {mount_path}")
        optical_path = mount_path + "/WristDepthOpticalFrame"
        optical = UsdGeom.Xform.Define(stage, optical_path)
        xform = UsdGeom.Xformable(optical)
        xform.ClearXformOpOrder()
        # RealSense nominal body(+X forward) -> ROS optical(+Z forward).
        # Quaternion for URDF rpy(-pi/2,0,-pi/2), zero translation.
        xform.AddOrientOp().Set(Gf.Quatf(.5, Gf.Vec3f(-.5, .5, -.5)))
        camera_path = optical_path + "/TrialWristRgbDepthCamera"
        author = RtxCamera(camera_path, tick_rate=30.)
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        camera.CreateFocalLengthAttr(12.)
        camera.CreateHorizontalApertureAttr(24. * math.tan(math.radians(87.) / 2))
        camera.CreateVerticalApertureAttr(24. * math.tan(math.radians(58.) / 2))
        # 0.07..0.5m is the maker's ideal range, not a hard hardware far limit.
        # Keep visible RGB surroundings to2m; measured depth is renderer ideal.
        camera.CreateClippingRangeAttr(Gf.Vec2f(.07, 2.))
        camera.CreateFocusDistanceAttr(.3)
        camera.GetPrim().SetCustomDataByKey("sg2_investigation", {
            "description": f"{side} wrist RGB + ideal image-plane depth",
            "mount_source": "ROBOTIS official SG2 URDF camera_left/right_link",
            "optical_source": "RealSense nominal D405 optical joints; zero offset",
            "calibration": "nominal FOV only; no measured intrinsics/stereo/noise",
        })
        set_usd_optical_orientation(stage, camera_path)
        records["wrist_" + side] = {
            "author": author, "mount_path": mount_path, "optical_path": optical_path,
            "sensor_path": camera_path, "copy_subtree_path": optical_path,
            "resolution_hw": [180, 320], "nominal_fov_deg": [87., 58.],
            "nominal_ideal_range_m": [.07, .5], "renderer_clip_m": [.07, 2.],
            "mount_offset_m": [0., 0., 0.],
        }
    return records


def set_usd_optical_orientation(stage, camera_path):
    from pxr import UsdGeom

    # CameraSensor initialization resets its Camera wrapper transform. Call
    # this AFTER wrapping as well to preserve ROS-optical->USD conversion.
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
    xform.ClearXformOpOrder()
    attribute = xform.GetPrim().GetAttribute("xformOp:rotateX")
    if attribute:
        attribute.Set(180.)
        xform.SetXformOpOrder([UsdGeom.XformOp(attribute)])
    else:
        xform.AddRotateXOp().Set(180.)
