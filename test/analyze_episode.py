#!/usr/bin/env python3
"""Analyse one collected episode (data.csv) to diagnose "spinning in place".

Usage:
    python analyze_episode.py <episode_dir_or_data.csv> [--tail N]

Prints:
  * cadence / duration / rows
  * 5 Hz macro_correction_type distribution (PASS / NORMAL / TURN_L/R)
  * hierarchical_mode / fsm_state distributions
  * command yaw-rate and speed statistics (spinning signals)
  * goal_distance_norm over time (start / min / end)
  * position footprint (is the drone actually translating?)
  * the last N rows' fsm_state / yaw_rate / speed / goal_dist sequence
"""
import csv
import math
import os
import sys


def _f(row, key, dflt=0.0):
    try:
        return float(row.get(key, dflt))
    except (TypeError, ValueError):
        return dflt


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    target = sys.argv[1]
    tail = 40
    for i, a in enumerate(sys.argv):
        if a == "--tail" and i + 1 < len(sys.argv):
            tail = int(sys.argv[i + 1])
    if os.path.isdir(target):
        csv_path = os.path.join(target, "data.csv")
    else:
        csv_path = target
    if not os.path.isfile(csv_path):
        print("no data.csv at %s" % csv_path)
        return 1

    rows = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    n = len(rows)
    if n == 0:
        print("empty csv")
        return 1
    print("=== %s ===" % csv_path)
    print("rows=%d  duration_s=%.1f  (30 Hz)" % (n, n / 30.0))

    # 5 Hz macro distribution (macro_update_mask==1 rows)
    macro = {}
    for r in rows:
        if r.get("macro_update_mask") == "1":
            ct = r.get("macro_correction_type", "?")
            macro[ct] = macro.get(ct, 0) + 1
    print("\n-- 5 Hz macro_correction_type (mask==1 rows) --")
    for k in sorted(macro):
        print("   %-18s %6d  (%.1f%%)" %
              (k, macro[k], 100.0 * macro[k] / max(1, sum(macro.values()))))

    # mode / fsm
    def _dist(key):
        d = {}
        for r in rows:
            v = r.get(key, "?")
            d[v] = d.get(v, 0) + 1
        return d

    print("\n-- hierarchical_mode --")
    for k, v in sorted(_dist("hierarchical_mode").items(),
                       key=lambda kv: -kv[1]):
        print("   %-16s %6d  (%.1f%%)" % (k, v, 100.0 * v / n))
    print("-- fsm_state --")
    for k, v in sorted(_dist("fsm_state").items(), key=lambda kv: -kv[1]):
        print("   %-16s %6d  (%.1f%%)" % (k, v, 100.0 * v / n))

    # command yaw rate / speed
    yaw_abs = [abs(_f(r, "target_yaw_rate")) for r in rows]
    yaw_abs.sort()
    speed = [math.hypot(_f(r, "target_velocity_flu_x"),
                        _f(r, "target_velocity_flu_y")) for r in rows]
    speed.sort()
    print("\n-- command |yaw_rate| (rad/s) --")
    for p in (0.5, 0.9, 0.99):
        print("   p%02d = %.3f" % (int(p * 100), yaw_abs[int(p * (n - 1))]))
    print("   mean = %.3f" % (sum(yaw_abs) / n))
    print("-- command speed (m/s) --")
    for p in (0.5, 0.9, 0.99):
        print("   p%02d = %.3f" % (int(p * 100), speed[int(p * (n - 1))]))
    print("   mean = %.3f" % (sum(speed) / n))
    slow = sum(1 for s in speed if s < 0.05)
    print("   fraction speed<0.05 = %.1f%%" % (100.0 * slow / n))

    # goal distance norm
    gd = [_f(r, "goal_distance_norm") for r in rows]
    print("\n-- goal_distance_norm --")
    print("   first=%.3f min=%.3f last=%.3f mean=%.3f" %
          (gd[0], min(gd), gd[-1], sum(gd) / n))

    # position footprint (translate or spin in place?)
    xs = [_f(r, "x") for r in rows]
    ys = [_f(r, "y") for r in rows]
    print("-- position footprint (m) --")
    print("   x range [%.2f, %.2f] span=%.2f" %
          (min(xs), max(xs), max(xs) - min(xs)))
    print("   y range [%.2f, %.2f] span=%.2f" %
          (min(ys), max(ys), max(ys) - min(ys)))
    # total path length
    path = 0.0
    for i in range(1, n):
        path += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
    print("   total path length = %.1f m" % path)

    # tail sequence
    print("\n-- last %d rows (idx | fsm | mode | yaw_rate | speed | goal_norm) --"
          % tail)
    for i in range(max(0, n - tail), n):
        r = rows[i]
        sp = math.hypot(_f(r, "target_velocity_flu_x"),
                        _f(r, "target_velocity_flu_y"))
        print("   %5d | %-14s | %-14s | %+.2f | %.2f | %.2f" %
              (i, r.get("fsm_state", "?")[:14],
               r.get("hierarchical_mode", "?")[:14],
               _f(r, "target_yaw_rate"), sp, _f(r, "goal_distance_norm")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
