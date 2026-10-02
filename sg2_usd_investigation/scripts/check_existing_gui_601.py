"""Bounded headless check of the Script Editor async lifecycle; no desktop window."""
import runpy
import time
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "enable_audio": False, "enable_motion_bvh": True,
                     "disable_viewport_updates": False})
scope = runpy.run_path(str(Path(__file__).with_name("open_sg2_in_existing.py")),
                      init_globals={"SG2_DEMO_ONCE": True, "SG2_EXIT_AFTER_DEMO": True})
task = scope["_sg2_task"]
deadline = time.monotonic() + 240
while task and not task.done() and time.monotonic() < deadline:
    app.update()
result = scope["result"]
ok = bool(task and task.done() and result.get("status") == "demo_complete_controls_ready"
          and result.get("assets_unchanged") and not result.get("sensor_errors")
          and len(result.get("wrist_camera_paths", {})) == 2)
print("EXISTING_GUI_ASYNC_RESULT:", result.get("status"), "passed=", ok, flush=True)
app.close(exit_code=0 if ok else 1)
