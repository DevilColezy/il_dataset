#!/usr/bin/env python3
"""debug_viewer.py  —  post-hoc review of the macro expert's behaviour.

Two input modes:

1. TRACE mode (default): reads `trace.jsonl` written per episode when
   `dataset_logging.debug_trace: true` is set.  Each line is one 5 Hz
   macro tick with the expert's full internal state (mode, reason, side,
   causal evidence, blocker tracking, candidates, recoverability audit).

2. CSV fallback mode (--csv): if no trace.jsonl exists, reads `data.csv`
   and reconstructs the macro timeline from the recorded macro_* columns
   (macro_is_new_tick frames only).

Usage:
    python debug_viewer.py <episode_dir_or_trace.jsonl> [--plot] [--csv] \
        [--limit N] [--mode MODE_NAME] [--show-candidates]

Examples:
    python debug_viewer.py dataset/il_data/_inprogress/ep1.inprogress
    python debug_viewer.py dataset/il_data/_inprogress/ep1.inprogress/trace.jsonl
    python debug_viewer.py dataset/il_data/_inprogress/ep1.inprogress --plot
    python debug_viewer.py dataset/il_data/ep1 --csv --limit 40

Exit codes: 0 = ok, 1 = no data found.
"""

from __future__ import print_function, division

import argparse
import csv
import json
import os
import sys

# Stable enum names (mirror types.hpp / il_macro_expert.SideFailure).
_MODE_NAMES = {0: "DIRECT_GUIDE", 1: "SIDE_GUIDE", 2: "OBSERVE",
               3: "GOAL_REACHED", 4: "FAILED"}
_SIDE_NAMES = {0: "NONE", 1: "LEFT", -1: "RIGHT"}
_REC_STATUS_NAMES = {0: "DIRECT_REJOIN_SUCCESS", 1: "PARTIAL_PROGRESS_ONLY",
                     2: "BLOCKED_BY_KNOWN", 3: "BLOCKED_BY_UNKNOWN",
                     4: "NO_SAFE_MOTION"}
_CAND_TYPE_NAMES = {0: "DIRECT", 1: "SIDE", 2: "OBSERVE",
                    3: "GOAL_FRONTIER", 4: "PREVIOUS_CONT"}


def _load_trace(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                print("[debug_viewer] skipping bad trace line: %s" % exc,
                      file=sys.stderr)
    return rows


def _load_csv_macro(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("macro_is_new_tick") == "1":
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def _pick_trace_target(target):
    """Return (path, mode) where mode is 'trace' or 'csv'."""
    if os.path.isdir(target):
        trace = os.path.join(target, "trace.jsonl")
        if os.path.isfile(trace):
            return trace, "trace"
        csv_path = os.path.join(target, "data.csv")
        if os.path.isfile(csv_path):
            return csv_path, "csv"
        return None, None
    if os.path.isfile(target):
        return target, "trace"
    return None, None


def _fmt_bool(value):
    return "yes" if str(value).lower() in ("1", "true", "yes") else "no"


def _print_trace_timeline(rows, limit, mode_filter, show_candidates):
    print("== Macro expert trace: %d ticks ==" % len(rows))
    printed = 0
    prev_mode = None
    for i, r in enumerate(rows):
        mode = r.get("mode_name") or _MODE_NAMES.get(
            int(r.get("mode", -1)), "?")
        if mode_filter and mode != mode_filter:
            continue
        t = r.get("trajectory_time_s", r.get("t_s", ""))
        frame = r.get("frame", "")
        reason = r.get("reason", "")
        side = r.get("committed_side", "")
        obs_side = r.get("observe_side", "")
        subtype = r.get("observe_subtype", "")
        conf = r.get("confidence", "")
        evidence = _fmt_bool(r.get("causal_intervention_evidence", 0))
        stable = r.get("direct_stable_ticks", "")
        released = _fmt_bool(r.get("blocker_released_this_tick", 0))
        failed_l = r.get("failed_left", "")
        failed_r = r.get("failed_right", "")
        blk = r.get("blocker_track_id", "")
        rec = r.get("rec_status", "")
        rec_name = _REC_STATUS_NAMES.get(int(rec), "?") if rec != "" else ""

        changed = "" if mode == prev_mode else " *"
        prev_mode = mode
        print("  [%3s] t=%s frame=%s  %-13s side=%-5s obs=%s/%s conf=%s%s"
              % (i, t, frame, mode, side, obs_side, subtype, conf, changed))
        print("        reason=%s  evidence=%s stable=%s released=%s"
              % (reason, evidence, stable, released))
        print("        failed_L=%s failed_R=%s blocker=%s rec=%s"
              % (failed_l, failed_r, blk, rec_name))
        if show_candidates and r.get("candidates"):
            names = []
            for c in r["candidates"][:5]:
                names.append("%s/%s(%.2f)" % (
                    _CAND_TYPE_NAMES.get(int(c.get("type", -1)), "?"),
                    c.get("side", "?"), c.get("score", 0.0)))
            print("        cands: %s" % ", ".join(names))
        printed += 1
        if limit and printed >= limit:
            print("  ... (truncated at %d rows)" % limit)
            break


def _print_transitions(rows):
    print("\n== Mode transitions ==")
    prev_mode = None
    for r in rows:
        mode = r.get("mode_name") or _MODE_NAMES.get(
            int(r.get("mode", -1)), "?")
        if mode != prev_mode:
            t = r.get("trajectory_time_s", r.get("t_s", ""))
            print("  t=%s  %-13s  reason=%s  side=%s"
                  % (t, mode, r.get("reason", ""), r.get("committed_side", "")))
            prev_mode = mode


def _print_csv_timeline(rows, limit):
    print("== Macro timeline from data.csv: %d macro ticks ==" % len(rows))
    printed = 0
    prev_mode = None
    for i, row in enumerate(rows):
        mode = _MODE_NAMES.get(int(row.get("macro_mode", -1)), "?")
        t = row.get("trajectory_time_s", "")
        side = _SIDE_NAMES.get(int(row.get("macro_committed_side", 0)), "?")
        reason = row.get("macro_decision_reason", "")
        changed = "" if mode == prev_mode else " *"
        prev_mode = mode
        print("  [%3s] t=%s  %-13s side=%-5s conf=%s%s"
              % (i, t, mode, side, row.get("macro_confidence", ""), changed))
        print("        reason=%s evidence=%s direct_no_progress=%s"
              % (reason,
                 _fmt_bool(row.get("causal_intervention_evidence", 0)),
                 row.get("direct_no_progress_time", "")))
        print("        observe_no_info=%s blocker=%s local_rec=%s"
              % (row.get("observe_no_information_time", ""),
                 row.get("blocker_track_id", ""),
                 row.get("local_recoverable", "")))
        printed += 1
        if limit and printed >= limit:
            print("  ... (truncated at %d rows)" % limit)
            break


def _plot_timeline(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[debug_viewer] matplotlib not installed; skipping plot.",
              file=sys.stderr)
        return
    times = []
    modes = []
    colors = []
    mode_color = {0: "tab:blue", 1: "tab:orange", 2: "tab:green",
                  3: "tab:red", 4: "tab:purple"}
    for r in rows:
        t = r.get("trajectory_time_s", r.get("t_s"))
        mode = r.get("mode") if "mode" in r else r.get("macro_mode")
        try:
            times.append(float(t))
            modes.append(int(mode))
            colors.append(mode_color.get(int(mode), "gray"))
        except (TypeError, ValueError):
            continue
    if not times:
        print("[debug_viewer] no plottable rows.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(times, modes, where="post", color="0.3", linewidth=1.0)
    ax.scatter(times, modes, c=colors, s=40, zorder=3)
    ax.set_yticks(sorted(_MODE_NAMES.keys()))
    ax.set_yticklabels([_MODE_NAMES[k] for k in sorted(_MODE_NAMES.keys())])
    ax.set_xlabel("trajectory time (s)")
    ax.set_ylabel("macro mode")
    ax.set_title("Macro expert mode timeline")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png)
        print("[debug_viewer] plot saved to %s" % out_png)
    else:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="View the macro expert's step-by-step behaviour.")
    parser.add_argument("target", help="episode dir or trace.jsonl path")
    parser.add_argument("--plot", action="store_true",
                        help="render a mode timeline figure")
    parser.add_argument("--plot-out", default="", help="save plot to this PNG")
    parser.add_argument("--csv", action="store_true",
                        help="force data.csv mode (ignore trace.jsonl)")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit printed ticks")
    parser.add_argument("--mode", default="",
                        help="only print ticks of this mode name")
    parser.add_argument("--show-candidates", action="store_true",
                        help="print top candidate summary per tick")
    args = parser.parse_args(argv)

    path, mode = _pick_trace_target(args.target)
    if path is None:
        print("[debug_viewer] no trace.jsonl or data.csv found under: %s"
              % args.target, file=sys.stderr)
        return 1
    if args.csv:
        mode = "csv"
        path = os.path.join(args.target, "data.csv") if os.path.isdir(
            args.target) else args.target

    if mode == "trace":
        rows = _load_trace(path)
        if not rows:
            print("[debug_viewer] trace.jsonl is empty: %s" % path,
                  file=sys.stderr)
            return 1
        _print_trace_timeline(rows, args.limit, args.mode,
                              args.show_candidates)
        _print_transitions(rows)
        if args.plot:
            _plot_timeline(rows, args.plot_out)
    else:
        rows = _load_csv_macro(path, args.limit)
        if not rows:
            print("[debug_viewer] no macro ticks in data.csv: %s" % path,
                  file=sys.stderr)
            return 1
        _print_csv_timeline(rows, args.limit)
        if args.plot:
            _plot_timeline(rows, args.plot_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
