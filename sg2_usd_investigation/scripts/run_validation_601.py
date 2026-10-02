"""Serial, bounded local GPU probes. No installation, source writes, commits or network publishing.

All simulator stdout/stderr are retained next to the structured result files.
"""
import json
import argparse
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
PYTHON = "/home/robotis/isaacsim/python.sh"
SCRIPTS = HERE / "scripts"
RESULTS = HERE / "results"
tasks = [
    ("original_motor_601", [str(SCRIPTS / "physics_probe_601.py"), "--case", "original_motor",
      "--usd", str(REPO / "source/cyclo_lab/data/robots/FFW/FFW_SG2.usd"),
      "--out", str(RESULTS / "physics_original_motor_601.json")]),
    ("cylinder_601", [str(SCRIPTS / "physics_probe_601.py"), "--case", "trial",
      "--usd", str(HERE / "runtime/FFW_SG2_cylinder.usd"),
      "--out", str(RESULTS / "physics_cylinder_601.json")]),
    ("sensors_601", [str(SCRIPTS / "sensor_probe_601.py"),
      "--usd", str(HERE / "runtime/FFW_SG2_trial.usd"),
      "--out-dir", str(RESULTS / "sensors_601"), "--frames", "180"]),
]
parser = argparse.ArgumentParser()
parser.add_argument("--only", choices=["matrix", "final", "gripper_off", "gripper_passive", "sensors", "sensors_wrist"], default="matrix")
args = parser.parse_args()
if args.only == "final":
    tasks = [
      ("final_trial_601", [str(SCRIPTS / "physics_probe_601.py"), "--case", "trial",
       "--usd", str(HERE / "runtime/FFW_SG2_trial.usd"), "--out", str(RESULTS / "physics_final_601.json")]),
      ("sensors_final_601", [str(SCRIPTS / "sensor_probe_601.py"), "--usd", str(HERE / "runtime/FFW_SG2_trial.usd"),
       "--out-dir", str(RESULTS / "sensors_final_601"), "--frames", "180"]),
    ]
elif args.only == "gripper_off":
    tasks = [("original_gripper_off_601", [str(SCRIPTS / "physics_probe_601.py"), "--case", "original_motor",
      "--self-collision", "off", "--usd", str(REPO / "source/cyclo_lab/data/robots/FFW/FFW_SG2.usd"),
      "--out", str(RESULTS / "physics_original_gripper_off_601.json")])]
elif args.only == "sensors":
    tasks = [("sensors_final_retry_601", [str(SCRIPTS / "sensor_probe_601.py"),
      "--usd", str(HERE / "runtime/FFW_SG2_trial.usd"),
      "--out-dir", str(RESULTS / "sensors_verified_601"), "--frames", "180"])]
elif args.only == "gripper_passive":
    tasks = [("original_gripper_passive_601", [str(SCRIPTS / "physics_probe_601.py"), "--case", "original_motor",
      "--slave-mode", "passive", "--usd", str(REPO / "source/cyclo_lab/data/robots/FFW/FFW_SG2.usd"),
      "--out", str(RESULTS / "physics_original_gripper_passive_601.json")])]
elif args.only == "sensors_wrist":
    tasks = [("sensors_wrist_601", [str(SCRIPTS / "sensor_probe_601.py"),
      "--usd", str(HERE / "runtime/FFW_SG2_trial.usd"),
      "--out-dir", str(RESULTS / "sensors_wrist_601"), "--frames", "180"])]
records = []
for name, arguments in tasks:
    command = [PYTHON, *arguments, "--/log/file="+str(RESULTS / (name+".kit.log"))]
    print("[VALIDATION] starting "+name, flush=True)
    start = time.monotonic()
    with (RESULTS / (name+".stdout.log")).open("w") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=420)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = "timeout_420s"
    records.append({"name":name,"command":command,"returncode":code,"elapsed_s":time.monotonic()-start})
    (RESULTS / f"validation_runs_{args.only}_601.json").write_text(json.dumps(records, indent=2)+"\n")
    print(f"[VALIDATION] {name} returncode={code}", flush=True)
print("[VALIDATION] complete; inspect JSON checks and retained logs", flush=True)
