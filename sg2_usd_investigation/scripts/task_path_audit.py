"""Inspect composed robot paths, colliders and references without starting Kit."""
import argparse
import ast
import json
import math
from pathlib import Path
from pxr import Usd, UsdGeom

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
folder = Path(__file__).resolve().parents[1]
repo = folder.parent
cfg = repo / "source/cyclo_lab/cyclo_lab/manager_based/manipulation/pick_place/config/ffw_sg2/joint_pos_env_cfg.py"
paths = sorted({node.value for node in ast.walk(ast.parse(cfg.read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith("{ENV_REGEX_NS}/Robot/")})
stage = Usd.Stage.CreateInMemory()
root_path = "/World/envs/env_0/Robot"
UsdGeom.Xform.Define(stage, root_path).GetPrim().GetReferences().AddReference(str(folder / "runtime/FFW_SG2_with_sensors.usda"))
checks = []
for path in paths:
    concrete = path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    # cam_head is spawned by CameraCfg. Its parent must exist, not the new camera.
    required = concrete.rsplit("/", 1)[0] if concrete.endswith("/cam_head") else concrete
    checks.append({"configured_path": path, "required_existing_path": required,
                   "exists_in_candidate": bool(stage.GetPrimAtPath(required))})
prims = list(stage.Traverse())
cameras = [str(p.GetPath()) for p in prims if p.IsA(UsdGeom.Camera)]
lidars = [str(p.GetPath()) for p in prims if p.GetTypeName() == "OmniLidar"]
wheel_shapes = []
for p in prims:
    if p.IsA(UsdGeom.Cylinder) and "wheel_drive_link" in str(p.GetPath()):
        geom = UsdGeom.Cylinder(p)
        wheel_shapes.append({"path": str(p.GetPath()), "radius_m": geom.GetRadiusAttr().Get(),
                             "width_m": geom.GetHeightAttr().Get(),
                             "matches_urdf_dimensions": math.isclose(geom.GetRadiusAttr().Get(), .0865, abs_tol=1e-6)
                             and math.isclose(geom.GetHeightAttr().Get(), .05, abs_tol=1e-6)})
layers = [str(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath]
result = {"scope": "static composed USD; no simulation or full task execution", "task_paths": checks,
          "drop_in_task_paths_pass": all(c["exists_in_candidate"] for c in checks),
          "camera_paths": cameras, "lidar_paths": lidars, "wheel_shapes": wheel_shapes,
          "sensor_counts_pass": len(cameras) == 4 and len(lidars) == 2,
          "loaded_layers": layers,
          "layers_inside_investigation": all(Path(p).is_relative_to(folder) for p in layers)}
args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"task_paths_pass": result["drop_in_task_paths_pass"],
                  "missing_paths": [c["configured_path"] for c in checks if not c["exists_in_candidate"]],
                  "camera_count": len(cameras), "lidar_count": len(lidars)}, ensure_ascii=False))
