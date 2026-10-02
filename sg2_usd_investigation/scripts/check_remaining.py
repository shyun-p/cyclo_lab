"""Bundle retained simulator evidence and fresh integration/dependency checks.

Does not launch Sim, install dependencies, publish, or modify official files.
Container config verification remains a separate bounded probe. Its results
are explicitly scoped to5.1, not treated as6.0 or full task validation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

folder = Path(__file__).resolve().parents[1]
repo = folder.parent
out = folder / "results"
python = "/home/robotis/isaacsim/python.sh"
usd_lib = Path("/home/robotis/isaacsim/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312")
physx_lib = Path("/home/robotis/isaacsim/extscache/omni.usd.schema.physx-110.1.13+110.1.2.lx64.r.cp312.u7f4")
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(map(str, (usd_lib, physx_lib)))
env["LD_LIBRARY_PATH"] = os.pathsep.join((str(usd_lib / "bin"), str(physx_lib / "bin"), env.get("LD_LIBRARY_PATH", "")))
env["PXR_PLUGINPATH_NAME"] = os.pathsep.join((str(physx_lib / "plugins/PhysxSchema/resources"), str(physx_lib / "plugins/PhysxSchemaAddition/resources")))
paths_run = subprocess.run([python, str(folder / "scripts/task_path_audit.py"), "--out", str(out / "task_path_audit.json")],
                           env=env, capture_output=True, text=True, timeout=60)
(out / "task_path_audit.stdout.log").write_text(paths_run.stdout + paths_run.stderr)
paths_run.check_returncode()
path_checks = json.loads((out / "task_path_audit.json").read_text())
dependencies = subprocess.run([python, "-c", 'import sys,importlib.util,json; print(json.dumps({"python":sys.version,"packages":{n:bool(importlib.util.find_spec(n)) for n in ("torch","gymnasium","isaaclab")}}))'],
                              capture_output=True, text=True, timeout=30)
dependencies.check_returncode()
deps = json.loads(dependencies.stdout)
def read(name):
    return json.loads((out / name).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
candidate_sha = sha(folder / "runtime/FFW_SG2_trial.usd")
static, physics, sensors = read("static_final_audit.json"), read("physics_final_601.json"), read("sensors_wrist_601/sensor_result.json")
baseline = read("baseline.json")
preserved = all(sha(repo / p) == digest for p, digest in baseline["official_sha256"].items())
cfg_path = out / "lab_config_51.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
result = {
    "scope": "bundled software readiness; hardware values need source data",
    "candidate_sha256": candidate_sha,
    "official_hashes_unchanged": preserved,
    "tracked_diff_empty": subprocess.run(["git", "diff", "--quiet"], cwd=repo).returncode == 0,
    "staged_diff_empty": subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0,
    "retained_evidence": {
        "static_39_link_match": static["candidate_vs_current_urdf"]["match_counts"],
        "static_matches_current_candidate": static["sources"]["candidate"]["sha256"] == candidate_sha,
        "standalone_physics_601_pass": physics["passed"] and physics["usd_sha256"] == candidate_sha,
        "standalone_sensor_601_pass": sensors["pass"] and sensors["candidate_sha256_before"] == candidate_sha,
        "camera_count": len(sensors["camera"]), "lidar_count": len(sensors["lidar"]),
        "scope": "reuse prior passed checks only after confirming unchanged candidate hash",
    },
    "config_factory_51_pass": cfg.get("pass", False) and cfg.get("candidate_sha256") == candidate_sha,
    "local_601_dependencies": deps,
    "task_paths": path_checks,
    "full_isaaclab_task_601_verified": False,
    "full_isaaclab_task_51_verified": False,
    "hardware_verified": False,
    "merge_ready": False,
    "software_work_remaining": [
        "Provide a compatible Isaac Lab/PyTorch/Gymnasium environment for Isaac Sim6.0.1, or agree on supported5.1 target",
        "Adapt all eef/camera paths to the candidate USD structure in a separate task",
        "Implement and verify floating-base pose/velocity resets and prevent duplicate head cameras",
        "Connect mobile commands and gripper clamp; verify19-to25 action producer/policy/dataset compatibility",
        "Align lidar profile to the agreed specification and run full task reset/step/camera/action regression",
    ],
    "questions_for_robotis": [
        "Is the PR scope a separate mobile SG2 environment or a change to existing fixed pick_place, and which Sim/Lab versions are supported?",
        "Which SG2 hardware revision and URDF commit are approved; can you provide missing measured/CAD mass, CoM and inertia data?",
        "What are the wheel/steer motor model, reduction ratio, continuous/peak output torque, speed limits and recommended control parameters?",
        "What are the actual head/wrist camera and lidar models, mounting calibration, camera intrinsics and lidar scan/range/noise settings?",
        "What exactly is the gripper mismatch (issue/reproducer), and what are the expected opening width, direction, limits, force and speed?",
    ],
}
(out / "remaining_checks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"config_51_pass": result["config_factory_51_pass"], "task_paths_pass": path_checks["drop_in_task_paths_pass"],
                  "local_601_packages": deps["packages"], "merge_ready": False}, ensure_ascii=False, indent=2))
