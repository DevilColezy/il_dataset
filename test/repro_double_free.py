#!/usr/bin/env python3
"""Reproduce the FlightmareDynamicsBridge double-free in isolation.

Loads BOTH pybind modules in the same process (expert first, matching
il_manager import order), creates/deletes FlightmareDynamics bridges, and
reports whether the Eigen/Quadrotor double free reproduces.
"""
import os
import sys
import gc
import numpy as np

sys.path.insert(0, "/home/rgzn/flightmare_ws/src/il_dataset/scripts")

import _il_hierarchical_expert  # noqa: E402  (load expert FIRST, like il_manager)
import _flightmare_dynamics     # noqa: E402

print("both modules imported OK")

for trial in range(3):
    b = _flightmare_dynamics.FlightmareDynamics()
    ok = b.reset(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
                 np.zeros(3), np.zeros(3))
    print("trial %d reset=%s" % (trial, ok))
    del b
    gc.collect()
    print("trial %d deleted bridge, gc ok" % trial)

print("ALL OK - no double free in this minimal path")
os._exit(0)
