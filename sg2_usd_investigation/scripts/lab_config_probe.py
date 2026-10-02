"""Check real Isaac Lab config factories in an isolated 5.1 container.

Read-only repository mount. No scene stepping, camera rendering, policy,
robot communication, or full-task success is claimed by this probe.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(folder / "runtime"))
result = {"status": "starting", "scope": "real config import/factory/selection check; not full task execution",
          "packages": {name: bool(importlib.util.find_spec(name)) for name in ("torch", "gymnasium", "isaaclab")}}
app = None
try:
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=False, device="cpu")
    app = launcher.app
    import torch
    from pxr import Usd, UsdPhysics
    from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG
    from cyclo_lab.manager_based.manipulation.pick_place.config.ffw_sg2.joint_pos_env_cfg import FFWSG2PickPlaceEnvCfg
    from FFW_SG2_trial import make_trial_cfg
    from mobile_actions import make_mobile_actions_cfg, clamp_gripper_command, ORIGINAL_ACTION_NAMES

    result["isaac_sim_version"] = Path("/isaac-sim/VERSION").read_text().strip()
    result["isaac_lab_version"] = (folder.parent / "third_party/IsaacLab/VERSION").read_text().strip()
    result["torch_version"] = torch.__version__
    original_before = FFW_SG2_CFG.to_dict()
    trial = make_trial_cfg()
    result["factory_checks"] = {
        "new_cfg_object": trial is not FFW_SG2_CFG,
        "new_spawn_object": trial.spawn is not FFW_SG2_CFG.spawn,
        "new_actuator_objects": all(trial.actuators[k] is not FFW_SG2_CFG.actuators[k]
                                    for k in FFW_SG2_CFG.actuators),
        "original_unchanged": original_before == FFW_SG2_CFG.to_dict(),
        "gravity_enabled": trial.spawn.rigid_props.disable_gravity is False,
        "slave_pd_zero": trial.actuators["gripper_slave"].stiffness == trial.actuators["gripper_slave"].damping == 0,
        "wheel_groups_present": all(k in trial.actuators for k in ("wheel_steer", "wheel_drive")),
    }
    stage = Usd.Stage.Open(trial.spawn.usd_path)
    result["candidate_sha256"] = hashlib.sha256(Path(trial.spawn.usd_path).read_bytes()).hexdigest()
    result["joint_names"] = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    result["action_modes"] = {}
    for mode in ("record", "inference", "mimic_ik"):
        cfg = FFWSG2PickPlaceEnvCfg()
        cfg.init_action_cfg(mode)
        mobile = make_mobile_actions_cfg(cfg.actions)
        result["action_modes"][mode] = {
            "original_terms_copied": all(getattr(mobile, name) is not getattr(cfg.actions, name)
                                         for name in ORIGINAL_ACTION_NAMES),
            "drive_joints_exist": all(j in result["joint_names"] for j in mobile.wheel_drive_action.joint_names),
            "steer_joints_exist": all(j in result["joint_names"] for j in mobile.wheel_steer_action.joint_names),
            "drive_targets": mobile.wheel_drive_action.joint_names,
            "steer_targets": mobile.wheel_steer_action.joint_names,
        }
    value = torch.tensor([-1., .5, 2.])
    result["clamp_tensor_pass"] = torch.equal(clamp_gripper_command(value), torch.tensor([0., .5, 1.]))
    result["pass"] = (all(result["factory_checks"].values()) and result["clamp_tensor_pass"]
                      and all(all(v[k] for k in ("original_terms_copied", "drive_joints_exist", "steer_joints_exist"))
                              for v in result["action_modes"].values()))
    result["status"] = "passed" if result["pass"] else "failed"
except Exception:
    result.update(status="failed", **{"pass": False}, traceback=traceback.format_exc())
finally:
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("SG2_LAB_CONFIG_CHECK:", result["status"], flush=True)
    if app:
        app.close()
    sys.exit(0 if result.get("pass") else 1)
