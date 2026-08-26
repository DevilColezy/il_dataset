#!/usr/bin/env python3
"""Direct UnityBridge test: connect to AvoidBench, send poses, check whether
matching render frames come back.  Isolates Unity rendering vs manager logic.
"""
import os
import sys
import time

sys.path.insert(0, "/home/rgzn/flightmare_ws/src/il_dataset/scripts")

import il_config        # noqa: E402
import il_common        # noqa: E402

CONFIG = "/home/rgzn/flightmare_ws/src/il_dataset/config/il_dataset_joint_v2_config.yaml"
cfg = il_config.load_config(CONFIG)
g = cfg["global"]
depth_cfg = g.get("depth", {})

pub_port = int(g.get("pub_port", 10253))
sub_port = int(g.get("sub_port", 10254))
print("pub=%s sub=%s depth_cfg keys=%s" % (
    pub_port, sub_port, sorted(depth_cfg.keys())[:12]))

bridge = il_common.UnityBridge(pub_port, sub_port)
bridge.bind()

ok = bridge.connect_handshake(1, depth_cfg, timeout=10.0)
print("handshake ready:", ok)

for i in range(6):
    warm_id = 5000 + i
    veh = il_common.make_depth_vehicle([0.0, 0.0, 2.0], 0.0, depth_cfg)
    bridge.send_pose({"scene_id": 1, "vehicles": [veh], "objects": [],
                      "frame_id": warm_id})
    deadline = time.time() + 4.0
    got = None
    while time.time() < deadline:
        r = bridge.try_recv()
        if r is not None:
            got = r
            break
        time.sleep(0.02)
    if got:
        keys = sorted(got[0].keys())
        fid = got[0].get("frame_id")
        print("frame %d OK  frame_id=%r wanted=%d match=%s  keys=%s" % (
            i, fid, warm_id, fid == warm_id, keys[:10]))
    else:
        print("frame %d NO RESPONSE in 4s" % i)

bridge.close()
os._exit(0)
