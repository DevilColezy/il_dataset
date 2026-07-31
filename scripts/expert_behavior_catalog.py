#!/usr/bin/env python3
"""Validated scenario catalog and suite selection for expert review tools."""

from __future__ import print_function

import os

import yaml


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_vec3(value, field, owner):
    if (not isinstance(value, list) or len(value) != 3 or
            not all(_is_number(item) for item in value)):
        raise ValueError(
            "{} '{}' must be a three-element numeric list".format(
                field, owner))


def _validate_scenario(name, scenario):
    if not isinstance(scenario, dict):
        raise ValueError("Scenario '{}' must be a mapping".format(name))
    _validate_vec3(scenario.get("start"), "start", name)
    _validate_vec3(scenario.get("goal"), "goal", name)

    obstacles = scenario.get("obstacles")
    if not isinstance(obstacles, list) or not obstacles:
        raise ValueError(
            "Scenario '{}' must contain at least one obstacle".format(name))
    obstacle_ids = set()
    for index, obstacle in enumerate(obstacles):
        owner = "{}.obstacles[{}]".format(name, index)
        if not isinstance(obstacle, dict):
            raise ValueError("{} must be a mapping".format(owner))
        _validate_vec3(obstacle.get("center"), "center", owner)
        obstacle_id = str(obstacle.get("id", "")).strip()
        if not obstacle_id or obstacle_id in obstacle_ids:
            raise ValueError(
                "{} needs a unique, non-empty id".format(owner))
        obstacle_ids.add(obstacle_id)
        for field in ("radius_m", "height_m"):
            value = obstacle.get(field)
            if not _is_number(value) or float(value) <= 0.0:
                raise ValueError(
                    "{}.{} must be positive".format(owner, field))


def _suite_names(suite_name, suite, scenarios):
    if isinstance(suite, list):
        names = suite
    elif isinstance(suite, dict):
        names = suite.get("scenarios")
    else:
        names = None
    if not isinstance(names, list) or not names:
        raise ValueError(
            "Suite '{}' must contain a non-empty scenarios list".format(
                suite_name))
    unknown = [name for name in names if name not in scenarios]
    if unknown:
        raise ValueError(
            "Suite '{}' references unknown scenarios {}".format(
                suite_name, unknown))
    if len(set(names)) != len(names):
        raise ValueError(
            "Suite '{}' contains duplicate scenarios".format(suite_name))
    return list(names)


def load_scenario_catalog(path):
    """Load and fully validate a deterministic expert-scenario catalog."""
    resolved = os.path.abspath(os.path.expanduser(path))
    with open(resolved, "r") as stream:
        catalog = yaml.safe_load(stream) or {}
    if not isinstance(catalog, dict):
        raise ValueError(
            "Scenario catalog must be a mapping: {}".format(resolved))

    scenarios = catalog.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(
            "Scenario catalog contains no scenarios: {}".format(resolved))
    for name, scenario in scenarios.items():
        _validate_scenario(name, scenario)

    raw_suites = catalog.get("suites", {})
    if not isinstance(raw_suites, dict) or not raw_suites:
        raise ValueError(
            "Scenario catalog contains no suites: {}".format(resolved))
    conflicts = sorted(set(raw_suites).intersection(scenarios))
    if conflicts:
        raise ValueError(
            "Suite and scenario names must be distinct: {}".format(
                conflicts))
    suites = {}
    for name, suite in raw_suites.items():
        suites[name] = _suite_names(name, suite, scenarios)

    acceptance = catalog.get("acceptance", {})
    if not isinstance(acceptance, dict):
        raise ValueError("'acceptance' must be a mapping")
    return resolved, catalog, scenarios, suites, acceptance


def select_scenarios(selector, scenarios, suites):
    """Resolve one scenario, one suite, ``all``, or a comma-separated mix."""
    text = str(selector).strip()
    if not text:
        raise ValueError("Scenario/suite selection must not be empty")
    if text.lower() == "all":
        return list(scenarios.keys())

    selected = []
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        if token in suites:
            additions = suites[token]
        elif token in scenarios:
            additions = [token]
        else:
            raise ValueError(
                "Unknown scenario/suite '{}'; scenarios={}, suites={}".format(
                    token, list(scenarios.keys()), list(suites.keys())))
        for name in additions:
            if name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError("No scenarios selected")
    return selected
