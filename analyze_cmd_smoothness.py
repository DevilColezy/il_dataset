#!/usr/bin/env python3
"""Analyze command smoothness of all collected trajectories."""
import csv
import glob
import math
import os


def col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, 0.0)))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def flips(d):
    n = 0
    for i in range(1, len(d)):
        if d[i] * d[i - 1] < 0:
            n += 1
    return n


def main():
    files = sorted(glob.glob('joint_v2_*/data.csv') +
                   glob.glob('_failed/*/data.csv'))
    print('%-28s %6s %7s %7s %7s %7s %7s %7s %7s' % (
        'episode', 'ticks', 'dSpd', 'mxSpd', 'dYawR', 'mxYaw', 'flpSpd',
        'flpYaw', 'dAng'))
    for f in files:
        ep = os.path.basename(os.path.dirname(f))
        rows = list(csv.DictReader(open(f)))
        vx = col(rows, 'velocity_command_flu_x')
        vy = col(rows, 'velocity_command_flu_y')
        yr = col(rows, 'yaw_rate_command')
        spd = [math.hypot(a, b) for a, b in zip(vx, vy)]
        ds = [abs(spd[i + 1] - spd[i]) for i in range(len(spd) - 1)]
        dyr = [abs(yr[i + 1] - yr[i]) for i in range(len(yr) - 1)]
        da = []
        for i in range(len(vx) - 1):
            a1 = math.atan2(vy[i], vx[i])
            a2 = math.atan2(vy[i + 1], vx[i + 1])
            da.append(math.degrees(abs(((a2 - a1 + math.pi) % (2 * math.pi)) - math.pi)))
        fs = flips([spd[i + 1] - spd[i] for i in range(len(spd) - 1)])
        fy = flips([yr[i + 1] - yr[i] for i in range(len(yr) - 1)])
        print('%-28s %6d %7.3f %7.3f %7.3f %7.3f %7d %7d %7.1f' % (
            ep, len(rows),
            sum(ds) / len(ds) if ds else 0,
            max(ds) if ds else 0,
            sum(dyr) / len(dyr) if dyr else 0,
            max(dyr) if dyr else 0,
            fs, fy, sum(da) / len(da) if da else 0))


if __name__ == '__main__':
    main()
