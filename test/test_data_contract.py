#!/usr/bin/env python3
"""Validate the fixed data contract (30 Hz / 5 Hz + depth uint16 PNG).

1. il_dataset_writer metadata records the depth encoding contract and the
   parameterized control / macro cadence.
2. save_net dataloader strictly excludes _failed/_inprogress, only accepts
   committed && episode_valid==1 rows with contiguous episode_frame_index,
   and masks invalid depth pixels (0) to max range.
"""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_NET = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "save_net")
if _NET not in sys.path:
    sys.path.insert(0, _NET)

passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError("FAIL: " + name)
    passed += 1
    print("ok -", name)


# ════════════════════════════════════════════════════════════════════
# 1) writer metadata contract
# ════════════════════════════════════════════════════════════════════
import il_dataset_writer  # noqa: E402

with tempfile.TemporaryDirectory() as root:
    w = il_dataset_writer.DatasetWriter(
        {"schema_version": 25, "flush_interval_rows": 0},
        "ep_meta_000001", root, 3, 7, [0.0, 0.0, 2.0], [5.0, 0.0, 2.0],
        0.0, {"width": 640, "height": 480},
        control_hz=30.0, macro_update_hz=5.0)
    md = w._metadata
    check("metadata local_control_hz param", md["local_control_hz"] == 30.0)
    check("metadata macro_update_hz param", md["macro_update_hz"] == 5.0)
    dec = md["depth_encoding_contract"]
    check("depth format", dec["format"] == "uint16_png")
    check("depth png_mode", dec["png_mode"] == "I;16")
    check("depth encoding normalized", dec["encoding"] == "normalized_16bit")
    check("depth decode formula",
          "65535" in dec["decode_formula"])
    check("depth max_m", dec["max_m"] == 5.0)
    check("depth invalid value", dec["invalid_pixel_value"] == 0)
    check("depth valid range m", dec["valid_range_m"] == [0.0, 5.0])
    check("data_contract mentions 30/5",
          "local_control_hz" in md["data_contract"] and
          "macro_update_hz" in md["data_contract"])
    # metadata.json persisted on disk (initial snapshot before close)
    with open(os.path.join(root, "_inprogress", "ep_meta_000001.inprogress",
                           "metadata.json"), "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    check("metadata persisted to disk",
          on_disk["depth_encoding_contract"]["encoding"] ==
          "normalized_16bit")
    w.close()

# ════════════════════════════════════════════════════════════════════
# 2) dataloader strictness
# ════════════════════════════════════════════════════════════════════
import dataloader  # noqa: E402
from dataloader import V25SequenceDataset, discover_committed_episodes  # noqa: E402

STATE_FIELDS = dataloader.STATE_FIELDS
TARGET_FIELDS = dataloader.TARGET_FIELDS
STATE_FIELDS_5HZ = dataloader.STATE_FIELDS_5HZ
LABEL_FIELDS_5HZ = dataloader.LABEL_FIELDS_5HZ


def make_row(idx, episode_valid="1", macro_mask="0", frame_valid="1"):
    row = {name: "0.0" for name in STATE_FIELDS + TARGET_FIELDS +
           STATE_FIELDS_5HZ}
    row.update({
        "episode_frame_index": str(idx),
        "frame_valid": frame_valid,
        "episode_valid": episode_valid,
        "depth_file": "depth/%06d.png" % idx,
        "hierarchical_mode": "direct",
        "planner_status": "SAFE_PROGRESSING",
        "macro_update_mask": macro_mask,
        "macro_label_valid": "1",
        "macro_correction_type": "PASS_THROUGH",
        "macro_direction_token": "-1",
        "macro_direction_flu_x": "1.0",
        "macro_direction_flu_y": "0.0",
        "macro_direction_flu_z": "0.0",
        "macro_distance_norm": "0.0",
        "macro_param_valid": "0",
        "goal_direction_flu_x": "1.0",
        "navigation_goal_direction_flu_x": "1.0",
        "target_velocity_flu_x": "1.0",
    })
    return row


FIELDS = (list(STATE_FIELDS) + list(TARGET_FIELDS) + list(STATE_FIELDS_5HZ) +
          list(LABEL_FIELDS_5HZ) + ["episode_frame_index", "frame_valid",
                                    "episode_valid", "depth_file",
                                    "hierarchical_mode", "planner_status"])


def write_episode(root: Path, name: str, rows, pixels):
    ep = root / name
    (ep / "depth").mkdir(parents=True)
    with (ep / "data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for idx, px in pixels:
        cv2.imwrite(str(ep / "depth" / ("%06d.png" % idx)),
                    np.asarray(px, dtype=np.uint16))
    with (ep / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "episode_id": name, "scene_id": "scene_a", "task_id": "task_a",
            "schema_version": 25, "status": "committed",
            "quality_committed": True, "reached_goal": True,
            "rows_written": len(rows),
        }, f)
    return ep


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # committed episode: 3 rows, valid; depth pixels with an invalid 0.
    rows = [make_row(0), make_row(1), make_row(2)]
    # Normalized 16-bit encoding: pixel = depth_m/5*65535, so
    # 13107 -> 1.0 m, 26214 -> 2.0 m, ..., 65535 -> 5.0 m.
    pixels = [(0, [[0, 13107, 26214], [39321, 52428, 65535]]),
              (1, [[13107, 13107, 13107], [13107, 13107, 13107]]),
              (2, [[65535, 65535, 65535], [65535, 65535, 65535]])]
    write_episode(root, "ep_ok", rows, pixels)
    # _failed and _inprogress must be excluded even with valid-looking data.
    write_episode(root / "_failed", "ep_failed", rows, pixels)
    write_episode(root / "_inprogress", "ep_inprog", rows, pixels)

    episodes = discover_committed_episodes(root)
    check("only committed discovered (excludes _failed/_inprogress)",
          [e.episode_id for e in episodes] == ["ep_ok"])
    check("episode rows", episodes[0].rows == 3)

    # committed but episode_valid=0 row must be rejected loudly.
    bad_rows = [make_row(0), make_row(1, episode_valid="0"), make_row(2)]
    with tempfile.TemporaryDirectory() as bad_tmp:
        write_episode(Path(bad_tmp), "ep_bad_valid", bad_rows, pixels)
        try:
            discover_committed_episodes(bad_tmp)
            raise AssertionError("FAIL: episode_valid!=1 was accepted")
        except ValueError as exc:
            check("episode_valid!=1 raises", "episode_valid" in str(exc))

    # invalid depth pixel (0) is masked to max range; valid normalized.
    ds = V25SequenceDataset(episodes, sequence_length=2, burn_in=0,
                            stride=2, augment=False, stateful=True,
                            max_depth_m=5.0)
    batch = ds[0]
    depth = batch["depth"].numpy()  # [T,1,H,W]
    # frame0 pixels (max=5): 0 -> invalid -> 1.0 ; 13107 -> 1.0/5 = 0.2 ;
    # 26214 -> 0.4 ; 39321 -> 0.6 ; 52428 -> 0.8 ; 65535 -> 1.0
    d0 = depth[0, 0]
    check("invalid pixel masked to 1.0", d0[0, 0] == 1.0)
    check("valid 1.0m -> 0.2", abs(d0[0, 1] - 0.2) < 1e-5)
    check("valid 2.0m -> 0.4", abs(d0[0, 2] - 0.4) < 1e-5)
    check("valid 3.0m -> 0.6", abs(d0[1, 0] - 0.6) < 1e-5)
    check("valid 4.0m -> 0.8", abs(d0[1, 1] - 0.8) < 1e-5)
    check("valid 5.0m -> 1.0", abs(d0[1, 2] - 1.0) < 1e-5)

print("\nALL %d CHECKS PASSED" % passed)
