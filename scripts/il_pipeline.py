#!/usr/bin/env python3
"""
IL Dataset Pipeline  —  **DEPRECATED**  —  Backward-compatible wrapper
=========================================================================

This module is DEPRECATED.  The canonical data-collection entry point is

    il_manager.py  (launched via il_dataset_collect.launch)

This wrapper delegates to the validated manager. It deliberately has no
legacy collector fallback: initialization errors stop formal collection.

Migration guide:
  1. Update your config to the v2 schema (see config/il_dataset_config.yaml).
  2. Use `roslaunch il_dataset il_dataset_collect.launch` instead of
     running il_pipeline.py directly.
  3. The v1 fields `global.flight`, `global.connect_timeout`, and
     `scene.trajectories` are replaced by `global.control`,
     `global.fsm.connect_timeout`, and automatic pair generation
     via `global.start_goal`.

Usage (deprecated – works but will warn):
    roslaunch flightmare_dataset_tools collect_dataset.launch
"""

from __future__ import print_function, division

import os, sys

import rospy
import rospkg


# Add script dir to path for legacy imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


def main():
    rospy.init_node("il_dataset_pipeline", anonymous=False)

    rospy.logwarn("=" * 60)
    rospy.logwarn("  il_pipeline.py is DEPRECATED and will be removed.")
    rospy.logwarn("  Please migrate to il_manager.py:")
    rospy.logwarn("    roslaunch il_dataset il_dataset_collect.launch")
    rospy.logwarn("  See config/il_dataset_config.yaml for the v2 schema.")
    rospy.logwarn("=" * 60)

    # Try to load config via the new validated loader
    try:
        from il_config import load_config
        from il_manager import ILManager

        cfg = load_config()
        mgr = ILManager(cfg)
        mgr.run()
        return
    except Exception as exc:
        rospy.logfatal("Validated IL manager failed: %s", exc)
        raise

    # ── Legacy fallback ────────────────────────────────────────────
    rospy.logwarn("Using LEGACY ILDatasetCollector – this path will be removed.")
    # No executable legacy path is retained here.


if __name__ == "__main__":
    main()

