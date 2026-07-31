#!/usr/bin/env python3
"""Run deterministic expert scenarios sequentially and write a JSON report."""

from __future__ import print_function

import argparse
import csv
import datetime
import glob
import json
import math
import os
import subprocess
import sys

import yaml


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the deterministic expert avoidance regression suite.")
    parser.add_argument(
        "--scenarios", default="all",
        help="Comma-separated scenario names, or 'all' (catalog order).")
    parser.add_argument(
        "--scenario-file",
        default=os.path.join(
            PACKAGE_DIR, "config", "expert_behavior_scenarios.yaml"))
    parser.add_argument(
        "--config-file",
        default=os.path.join(PACKAGE_DIR, "config", "il_dataset_config.yaml"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PACKAGE_DIR, "dataset",
                             "expert_behavior_suite"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Open each interactive diagnostic (disabled by default).")
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Stop launching scenarios after the first failure.")
    return parser.parse_args()


def _load_catalog(path):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r") as stream:
        catalog = yaml.safe_load(stream) or {}
    scenarios = catalog.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("No scenarios found in {}".format(path))
    acceptance = catalog.get("acceptance", {})
    if not isinstance(acceptance, dict):
        raise ValueError("'acceptance' must be a mapping")
    return path, scenarios, acceptance


def _select_scenarios(spec, scenarios):
    if spec.strip().lower() == "all":
        return list(scenarios.keys())
    selected = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [item for item in selected if item not in scenarios]
    if unknown:
        raise ValueError(
            "Unknown scenarios {}; available: {}".format(
                unknown, list(scenarios.keys())))
    if not selected:
        raise ValueError("No scenarios selected")
    return selected


def _newest_metadata(run_dir):
    paths = glob.glob(
        os.path.join(run_dir, "**", "metadata.json"), recursive=True)
    if not paths:
        return None, None
    path = max(paths, key=os.path.getmtime)
    with open(path, "r") as stream:
        return path, json.load(stream)


def _peak_speed(trajectory_dir):
    data_path = os.path.join(trajectory_dir, "data.csv")
    if not os.path.isfile(data_path):
        return 0.0
    peak = 0.0
    with open(data_path, "r") as stream:
        for row in csv.DictReader(stream):
            try:
                values = [
                    float(row["state_vx_world"]),
                    float(row["state_vy_world"]),
                    float(row["state_vz_world"]),
                ]
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values):
                peak = max(peak, math.sqrt(sum(v * v for v in values)))
    return peak


def _evaluate(metadata_path, metadata, acceptance):
    failures = []
    if metadata is None:
        return {
            "passed": False,
            "failures": ["metadata_missing"],
            "metadata_path": None,
        }

    trajectory_dir = os.path.dirname(metadata_path)
    peak_speed = _peak_speed(trajectory_dir)
    if acceptance.get("require_reached_goal", True) and not bool(
            metadata.get("reached_goal", False)):
        failures.append("goal_not_reached")
    if acceptance.get("require_zero_invalid_frames", True) and int(
            metadata.get("invalid_frame_count", -1)) != 0:
        failures.append("invalid_frames={}".format(
            metadata.get("invalid_frame_count")))
    if acceptance.get("require_zero_planner_failures", True) and int(
            metadata.get("planner_failure_count", -1)) != 0:
        failures.append("planner_failures={}".format(
            metadata.get("planner_failure_count")))

    checks = [
        ("max_planning_ms", "maximum_planning_ms",
         33.3, "maximum", float("inf")),
        ("minimum_executed_clearance",
         "minimum_executed_clearance_m",
         0.02, "minimum", float("-inf")),
        ("final_goal_error", "maximum_final_goal_error_m",
         0.30, "maximum", float("inf")),
    ]
    for meta_key, criterion_key, default, direction, missing_value in checks:
        value = float(metadata.get(meta_key, missing_value))
        limit = float(acceptance.get(criterion_key, default))
        if ((direction == "maximum" and value > limit) or
                (direction == "minimum" and value < limit)):
            failures.append("{}={:.4f} {} {:.4f}".format(
                meta_key, value,
                ">" if direction == "maximum" else "<", limit))

    min_peak = float(acceptance.get("minimum_peak_speed_mps", 1.20))
    if peak_speed < min_peak:
        failures.append(
            "peak_speed_mps={:.4f} < {:.4f}".format(peak_speed, min_peak))

    return {
        "passed": not failures,
        "failures": failures,
        "metadata_path": metadata_path,
        "trajectory_dir": trajectory_dir,
        "status": metadata.get("status"),
        "reached_goal": metadata.get("reached_goal"),
        "invalid_frame_count": metadata.get("invalid_frame_count"),
        "planner_failure_count": metadata.get("planner_failure_count"),
        "max_planning_ms": metadata.get("max_planning_ms"),
        "minimum_executed_clearance":
            metadata.get("minimum_executed_clearance"),
        "final_goal_error": metadata.get("final_goal_error"),
        "peak_speed_mps": round(peak_speed, 4),
    }


def _print_result(name, result):
    outcome = "PASS" if result["passed"] else "FAIL"
    detail = ", ".join(result["failures"]) if result["failures"] else "ok"
    print("[{}] {:<28} {}".format(outcome, name, detail), flush=True)


def main():
    args = _parse_args()
    scenario_file, scenarios, acceptance = _load_catalog(
        args.scenario_file)
    selected = _select_scenarios(args.scenarios, scenarios)
    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_dir = os.path.join(output_root, stamp)
    os.makedirs(suite_dir)

    results = []
    for index, name in enumerate(selected):
        run_dir = os.path.join(suite_dir, name)
        command = [
            "roslaunch", "il_dataset", "expert_behavior_test.launch",
            "config_file:={}".format(
                os.path.abspath(os.path.expanduser(args.config_file))),
            "scenario:={}".format(name),
            "scenario_file:={}".format(scenario_file),
            "output_dir:={}".format(run_dir),
            "seed:={}".format(args.seed + index),
            "show_plot:={}".format(
                "true" if args.show_plot else "false"),
        ]
        print("\n[RUN] {}".format(name), flush=True)
        return_code = subprocess.call(command)
        metadata_path, metadata = _newest_metadata(run_dir)
        result = _evaluate(metadata_path, metadata, acceptance)
        result.update({
            "scenario": name,
            "description": scenarios[name].get("description", ""),
            "roslaunch_return_code": return_code,
        })
        if return_code != 0:
            result["passed"] = False
            result["failures"].append(
                "roslaunch_exit={}".format(return_code))
        results.append(result)
        _print_result(name, result)
        if args.stop_on_failure and not result["passed"]:
            break

    report = {
        "version": 1,
        "scenario_file": scenario_file,
        "config_file": os.path.abspath(os.path.expanduser(args.config_file)),
        "acceptance": acceptance,
        "selected_scenarios": selected,
        "passed": all(item["passed"] for item in results),
        "completed_count": len(results),
        "results": results,
    }
    report_path = os.path.join(suite_dir, "suite_report.json")
    with open(report_path, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print("\nSuite report: {}".format(report_path), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
