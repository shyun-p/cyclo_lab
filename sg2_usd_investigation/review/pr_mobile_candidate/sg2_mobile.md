# SG2 mobile configuration example

This optional configuration copies `FFW_SG2_CFG`. Existing robot settings and
pick_place tasks remain unchanged. Supply a floating-base SG2 USD and explicit
wheel parameters; this change contains no default wheel motor specification.

## Scope

The factory enables gravity, separates steering position control from wheel
velocity control, and disables competing PD control on mimic gripper joints.
The example creates a one-robot Isaac Lab scene and runs a bounded forward,
lateral, gripper and root-reset sequence. It does not connect to robot hardware.

The caller's USD must already contain a floating articulation, wheel colliders
and valid gripper mimic constraints. The example rejects enabled world fixed
joints. The supplied USD is referenced and never saved.

This is a configuration/example addition, not a complete correction of all
reported SG2 issues. It does not synchronize mass properties, generate a USD,
create calibrated sensors, adapt the full pick_place task or update policies and
datasets. The supplied USD's provenance must be checked independently.

## Run

Use a cyclo_lab installation with Isaac Lab initialized through its normal
launcher. The repository's baseline is Isaac Lab2.3 / Isaac Sim5.1.

```bash
python scripts/examples/ffw_sg2_mobile.py --headless --device cpu \
    --usd /absolute/path/to/floating_sg2.usd \
    --steer-stiffness 10000 --steer-damping 100 \
    --steer-effort 1000 --steer-speed-limit 30 \
    --drive-damping 100 --drive-effort 100 --drive-speed-limit 30 \
    --out /absolute/path/to/sg2_mobile_result.json
```

The numbers in this invocation are **provisional simulation test parameters**,
not ROBOTIS motor specifications. Gains are in Isaac Lab's SI convention;
angular speed is rad/s and angular effort is N·m. The local example floor uses
friction1, not a measured tire/floor coefficient. Hardware parameters and sensor
calibration require manufacturer data. The gripper example clamps its command
to0..1rad; applications must confirm the supplied USD's valid joint range.

The JSON records the actual parameters, USD hash, movement, gripper following
and original cfg preservation. Check `pass` and `usd_unchanged`, not only the
process exit code. These example checks are not a full task or hardware test.

For GUI use, omit `--headless`. Camera/LiDAR rendering is outside this example.
Isaac Sim6.0.1 requires its own compatible Isaac Lab/dependency environment;
passing this example under5.1 does not establish6.0.1 support.
