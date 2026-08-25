#!/usr/bin/env python3
"""Scan velocity-loop gains: command constant forward, measure the
body-forward speed oscillation amplitude (R29e 2.2 Hz limit-cycle check).

Usage: python3 verify_velocity_gains.py [kp] [kd] [cmd_speed]
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))
sys.path.insert(0, _HERE)

import il_config            # noqa: E402
import il_dynamics          # noqa: E402

CONFIG = os.path.join(_HERE, "config", "il_dataset_joint_v2_config.yaml")

# Keep every backend alive: the C++ bridge destructor double-frees, so we
# never let one go out of scope — hard-exit at the end instead.
_KEEP = []


def run(cmd_spd, dur_s=5.0, deriv_tau=None, control_hz=None,
        ang_rate_gain=None):
    cfg = il_config.load_config(CONFIG)
    if deriv_tau is not None:
        cfg["global"]["dynamics"]["velocity_controller"][
            "derivative_filter_tau_s"] = deriv_tau
    if control_hz is not None:
        cfg["global"]["dynamics"]["control_hz"] = control_hz
    if ang_rate_gain is not None:
        cfg["global"]["dynamics"]["velocity_controller"][
            "angular_rate_gain"] = ang_rate_gain
    d = il_dynamics.FlightmareDynamicsBackend(cfg)
    _KEEP.append(d)
    d.reset([0.0, 7.0, 2.0], 0.0)
    cmd = np.array([cmd_spd, 0.0, 0.0])
    n_steps = int(dur_s * 50)
    vf, rolls, pitches = [], [], []
    for _ in range(n_steps):
        d.step_velocity_command(cmd, 0.0, 1.0 / 50.0)
        s = d.get_state()
        v = np.asarray(s.velocity_world)
        q = np.asarray(s.quaternion_world_body)  # xyzw
        q /= max(float(np.linalg.norm(q)), 1e-12)
        x, y, z, w = q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, sn = math.cos(yaw), math.sin(yaw)
        # FM convention: forward=(-sin,cos), left=(-cos,-sin)
        vx_body = -sn * v[0] + c * v[1]
        vy_body = -c * v[0] - sn * v[1]
        vf.append(vx_body)
        # ZYX euler from xyzw: roll, pitch
        roll = math.degrees(math.atan2(2 * (w * x + y * z),
                                       1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1, min(
            1, 2 * (w * y - z * x)))))
        rolls.append(roll)
        pitches.append(pitch)
    w = vf[int(len(vf) * 0.3):]
    wr = rolls[int(len(rolls) * 0.3):]
    wp = pitches[int(len(pitches) * 0.3):]
    spd_amp = (max(w) - min(w)) / 2
    roll_amp = (max(wr) - min(wr)) / 2
    pitch_amp = (max(wp) - min(wp)) / 2
    return (sum(w) / len(w), spd_amp, roll_amp, pitch_amp)


def main():
    cmd_spd = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.6
    print("cmd=%.2f m/s forward, 5 s (control_hz=200);  steady 2 s:" % cmd_spd)
    print("  %12s | %8s %8s %8s %8s" % (
        "angRateGain", "meanV", "spdAmp", "rollAmp", "pitchAmp"))
    for g in (0.0, 0.3, 0.5, 0.8):
        mean, sa, ra, pa = run(cmd_spd, ang_rate_gain=g)
        print("  %12.1f | %8.3f %8.3f %8.3f %8.3f" % (
            g, mean, sa, ra, pa), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
