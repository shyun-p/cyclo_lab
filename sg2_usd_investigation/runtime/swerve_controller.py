"""Trial three-wheel swerve IK, using SI units and measured base heading.

Pure Python: the caller provides world velocity and measured yaw/steer angles.
This is simulation control code; the gains/geometry are not hardware certified.
"""
import math

WHEEL_POSITIONS_M = ((.1371, .2554), (.1371, -.2554), (-.2899, 0.))
WHEEL_RADIUS_M = .0865


def wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def wheel_targets(vx_world, vy_world, yaw_rad, heading_rad=0., current_steer=None,
                  yaw_gain=2., yaw_rate_limit=.5):
    """Return steer radians and drive rad/s for left, right, rear wheel.

    Hold heading with limited yaw velocity, transform world translation to body,
    then add rotational velocity at each wheel. Choose a nearby steering angle;
    reverse wheel speed when a half-turn would otherwise be required.
    """
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    vx, vy = cy*vx_world+sy*vy_world, -sy*vx_world+cy*vy_world
    omega = max(-yaw_rate_limit, min(yaw_rate_limit, yaw_gain*wrapped_angle(heading_rad-yaw_rad)))
    steer, drive = [], []
    current_steer = current_steer if current_steer is not None else (0., 0., 0.)
    for (x,y), current in zip(WHEEL_POSITIONS_M, current_steer):
        wx, wy = vx-omega*y, vy+omega*x
        speed = math.hypot(wx,wy)/WHEEL_RADIUS_M
        if speed < 1e-10:
            steer.append(current)
            drive.append(0.)
            continue
        delta = wrapped_angle(math.atan2(wy,wx)-current)
        if abs(delta) > math.pi/2:
            delta = wrapped_angle(delta-math.copysign(math.pi,delta))
            speed = -speed
        steer.append(current+delta)
        drive.append(speed)
    return steer, drive
