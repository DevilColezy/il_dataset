#!/usr/bin/env python3
"""Standalone validation of the il_manager committed-stats + top-up logic.

Mocks the ROS / pybind dependencies, then exercises the REAL methods of
JointV2Manager (statistics accumulation, gap evaluation, task scoring)
against hand-computed CSV data.  Run on Windows without ROS:
    python debug/test_topup_stats.py
"""
import math
import os
import sys
import tempfile
import types

# ── Mock ROS / project modules before importing il_manager ────────────
def _nolog(*a, **k):
    pass

rospy = types.ModuleType("rospy")
rospy.loginfo = _nolog
rospy.logwarn = _nolog
rospy.logerr = _nolog
rospy.logfatal = _nolog
rospy.init_node = _nolog
rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
rospy.is_shutdown = lambda: False
rospy.has_param = lambda *a: False
rospy.get_param = lambda *a, **k: None
sys.modules["rospy"] = rospy

il_config = types.ModuleType("il_config")
il_config.load_config = lambda *a, **k: {"global": {}}
sys.modules["il_config"] = il_config

il_common = types.ModuleType("il_common")

class UnityBridge(object):
    pass

il_common.UnityBridge = UnityBridge
il_common.world_vector_to_body_flu_quat = lambda *a, **k: (0.0, 0.0, 0.0)
sys.modules["il_common"] = il_common

il_expert_config = types.ModuleType("il_expert_config")
il_expert_config.build_params = lambda *a, **k: (None, [])
il_expert_config.build_scene_bounds = lambda *a, **k: ([0.0, 0.0], [1.0, 1.0])
sys.modules["il_expert_config"] = il_expert_config

il_dataset_writer = types.ModuleType("il_dataset_writer")

class DatasetWriter(object):
    pass

il_dataset_writer.DatasetWriter = DatasetWriter
sys.modules["il_dataset_writer"] = il_dataset_writer

expert_mod = types.ModuleType("_il_hierarchical_expert")

class HierarchicalExpert(object):
    pass

class TruthCylinderAudit(object):
    pass

class SceneTaskBlueprintGenerator(object):
    pass

expert_mod.HierarchicalExpert = HierarchicalExpert
expert_mod.TruthCylinderAudit = TruthCylinderAudit
expert_mod.SceneTaskBlueprintGenerator = SceneTaskBlueprintGenerator
sys.modules["_il_hierarchical_expert"] = expert_mod

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import il_manager  # noqa: E402

M = il_manager.JointV2Manager
passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError("FAIL: " + name)
    passed += 1
    print("ok -", name)


# ── Instance without __init__ (bypass ROS/config wiring) ─────────────
m = object.__new__(M)
m._g = {
    "blueprint_generation": {
        "task_generation": {"histograms": {}},
        "requirements": {"min_macro_ticks_per_class": 24},
        "topup": {"max_rounds": 3},
    },
}

# ── _distribution_cfg defaults ───────────────────────────────────────
cfg = m._distribution_cfg()
check("mtc default", cfg["mtc"] == 24)
check("deflection edges", cfg["deflection_edges"] ==
      [-90.0, -60.0, -30.0, -10.0, 10.0, 30.0, 60.0, 90.0])
check("correction edges", cfg["correction_angle_edges"][0] == -90.0 and
      cfg["correction_angle_edges"][-1] == 90.0 and
      len(cfg["correction_angle_edges"]) == 11)
check("speed edges", cfg["speed_edges"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
check("min deflection speed", abs(cfg["min_deflection_speed_mps"] - 0.10) < 1e-9)

# ── histogram binning (mirror of C++ Histogram1D::binOf) ─────────────
E = cfg["correction_angle_edges"]
check("bin <= first edge", M._hist_bin_of(E, -91.0) == 0)
check("bin == first edge", M._hist_bin_of(E, -90.0) == 0)
check("bin inside 1", M._hist_bin_of(E, -45.5) == 1)
check("bin inside 2", M._hist_bin_of(E, -45.0) == 2)
check("bin inside 3", M._hist_bin_of(E, 0.0) == 5)
check("bin inside 4", M._hist_bin_of(E, 30.0) == 7)
check("bin last edge", M._hist_bin_of(E, 90.0) == 9)
check("bin beyond last", M._hist_bin_of(E, 120.0) == 9)
check("bin nan", M._hist_bin_of(E, float("nan")) == -1)
check("bin empty", M._hist_bin_of([], 1.0) == -1)

# ── wrapAngle mirror ─────────────────────────────────────────────────
check("wrap 190", abs(M._wrap_angle_deg(190.0) - (-170.0)) < 1e-9)
check("wrap -190", abs(M._wrap_angle_deg(-190.0) - 170.0) < 1e-9)
check("wrap 180", abs(M._wrap_angle_deg(180.0) - 180.0) < 1e-9)
check("wrap -180", abs(M._wrap_angle_deg(-180.0) - (-180.0)) < 1e-9)
check("wrap 0", abs(M._wrap_angle_deg(0.0)) < 1e-9)

# ── committed-episode accumulation from a synthetic data.csv ─────────
# 2 5Hz rows (PASS, NORMAL), 3 30Hz rows.  NORMAL row:
#   nav goal dir = (1,0), effective target dir = (cos30, sin30)  -> +30 deg
#   macro_distance_norm = 0.5
# 30Hz rows: direct/avoidance, speeds, yaw rates, deflections.
rows = [
    # idx0: 30Hz, direct, speed 1.0, yaw_rate 0.1, deflection n/a
    dict(episode_valid="1", hierarchical_mode="direct",
         avoidance_active="0", target_velocity_flu_x="1.0",
         target_velocity_flu_y="0.0", target_yaw_rate="0.1",
         goal_direction_flu_x="1.0", goal_direction_flu_y="0.0",
         macro_update_mask="0", macro_correction_type="PASS_THROUGH",
         target_correction_active="0", navigation_goal_direction_flu_x="1.0",
         navigation_goal_direction_flu_y="0.0", macro_distance_norm="0.0"),
    # idx1: 30Hz + 5Hz PASS, avoidance, speed 0.3 (below min deflect),
    #       yaw_rate 0.2
    dict(episode_valid="1", hierarchical_mode="local_avoidance",
         avoidance_active="1", target_velocity_flu_x="0.3",
         target_velocity_flu_y="0.0", target_yaw_rate="0.2",
         goal_direction_flu_x="1.0", goal_direction_flu_y="0.0",
         macro_update_mask="1", macro_correction_type="PASS_THROUGH",
         target_correction_active="0", navigation_goal_direction_flu_x="1.0",
         navigation_goal_direction_flu_y="0.0", macro_distance_norm="0.0"),
    # idx2: 30Hz + 5Hz NORMAL + correction active:
    #   effective dir = (cos30, sin30) vs nav (1,0) -> +30 deg correction
    #   speed 1.5, yaw_rate -0.3; deflection = v(1.5,0) vs g(cos30,sin30)
    #     = -30 deg
    dict(episode_valid="1", hierarchical_mode="macro_normal",
         avoidance_active="1", target_velocity_flu_x="1.5",
         target_velocity_flu_y="0.0", target_yaw_rate="-0.3",
         goal_direction_flu_x="0.8660254037844386",
         goal_direction_flu_y="0.5",
         macro_update_mask="1", macro_correction_type="NORMAL_CORRECTION",
         target_correction_active="1", navigation_goal_direction_flu_x="1.0",
         navigation_goal_direction_flu_y="0.0", macro_distance_norm="0.5"),
    # idx3: 30Hz TURN_LEFT 5Hz, speed 0.8, yaw_rate 0.7 (turn)
    dict(episode_valid="1", hierarchical_mode="macro_turn_left",
         avoidance_active="0", target_velocity_flu_x="0.0",
         target_velocity_flu_y="0.8", target_yaw_rate="0.7",
         goal_direction_flu_x="0.0", goal_direction_flu_y="1.0",
         macro_update_mask="1", macro_correction_type="TURN_LEFT",
         target_correction_active="1", navigation_goal_direction_flu_x="1.0",
         navigation_goal_direction_flu_y="0.0", macro_distance_norm="1.0"),
]
# sanity: idx2 deflection = angle(g=(.866,.5), v=(1.5,0)) = atan2(cross,dot)
# cross = gx*vy - gy*vx = .866*0 - .5*1.5 = -0.75; dot = .866*1.5 = 1.299
# ang = atan2(-0.75, 1.299) = -30 deg -> deflection bin -30 (edges bin2 [-30,-10)? )
# v=-30: not <= -60, not >= 90; upper_bound first > -30 -> -10 (idx4)? edges
# [-90,-60,-30,-10,10,...]; first > -30 = -10 idx3 -> bin2 = [-30,-10). yes bin2.
# correction angle = angle(nav=(1,0), eff=(.866,.5)) = atan2(1*.5-0*.866, 1*.866)=atan2(.5,.866)=30 -> bin7 [30,45)
tmpdir = tempfile.mkdtemp(prefix="topup_test_")
episode_dir = os.path.join(tmpdir, "ep_committed")
os.makedirs(episode_dir)
with open(os.path.join(episode_dir, "data.csv"), "w", newline="") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
# also a rejected episode that must be ignored
rej_dir = os.path.join(tmpdir, "ep_rejected")
os.makedirs(rej_dir)
rej = dict(rows[0])
rej["episode_valid"] = "0"
with open(os.path.join(rej_dir, "data.csv"), "w", newline="") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=sorted(rej.keys()))
    w.writeheader()
    w.writerow(rej)

stats = m._new_committed_stats(cfg)
n = m._add_committed_episode(stats, episode_dir, cfg)
check("committed rows counted", n == 4)
check("rejected ignored", m._add_committed_episode(stats, rej_dir, cfg) == 0)

c = stats["counts"]
check("macro:total", c["macro:total"] == 3)
check("macro:pass", c["macro:pass"] == 1)
check("macro:normal", c["macro:normal"] == 1)
check("macro:turn_left", c["macro:turn_left"] == 1)
check("macro:turn_right", c["macro:turn_right"] == 0)
check("local:direct", c["local:direct"] == 1)
check("local:avoidance", c["local:avoidance"] == 2)

# speed bins: 1.0 -> [1.0,1.5)? edges[0,0.5,1,1.5,2,2.5,3]; 1.0 -> upper_bound
# first > 1.0 = 1.5 idx3 -> bin2 [1.0,1.5). 0.3->bin0, 1.5->bin3, 0.8->bin1
sp = stats["hists"]["local_speed"]
check("speed bin total", sum(sp) == 4)
check("speed bin0 (0.3)", sp[0] == 1)
check("speed bin1 (0.8)", sp[1] == 1)
check("speed bin2 (1.0)", sp[2] == 1)
check("speed bin3 (1.5)", sp[3] == 1)

# yaw_rate bins: 0.1->bin4 [0,0.2)? edges[-2,-1,-.5,-.2,0,.2,.5,1,2]; first>0.1=.2 idx5->bin4.
# 0.2->bin5, -0.3->bin3 [-.5,-.2)? upper_bound first>-0.3=-0.2 idx4->bin3. 0.7->bin6
yr = stats["hists"]["local_yaw_rate"]
check("yaw bin total", sum(yr) == 4)
check("yaw bin2 (-0.3)", yr[2] == 1)
check("yaw bin4 (0.1)", yr[4] == 1)
check("yaw bin5 (0.2)", yr[5] == 1)
check("yaw bin6 (0.7)", yr[6] == 1)

# deflection: all 4 rows have speed >= 0.1.  idx0/1/3 = 0deg -> bin3
# [-10,10); idx2 = -30deg -> bin2 [-30,-10).
df = stats["hists"]["local_deflection"]
check("deflection total", sum(df) == 4)
check("deflection bin2 (-30deg)", df[2] == 1)
check("deflection bin3 (0deg)", df[3] == 3)

# correction angle: only NORMAL active row -> +30 deg -> bin7 [30,45)
ca = stats["hists"]["macro_correction_angle"]
check("correction angle total", sum(ca) == 1)
check("correction angle bin7 (+30)", ca[7] == 1)

# correction distance: 0.5 -> bin2 [0.4,0.6)
cd = stats["hists"]["macro_correction_distance"]
check("correction distance total", sum(cd) == 1)
check("correction distance bin2 (0.5)", cd[2] == 1)

# ── gap evaluation ───────────────────────────────────────────────────
targets = m._gap_targets(cfg)
gaps = m._evaluate_gaps(stats, targets)
check("gap for macro:pass", gaps.get("macro:pass") is not None)
check("gap for macro:turn_left",
      abs(gaps["macro:turn_left"][1] - 72.0) < 1e-9)
check("no gap for macro:turn_right target shape",
      gaps["macro:turn_right"][2] > 0.0)  # 0 achieved vs 72 target
check("gap deficit positive", all(v[2] > 0.0 for v in gaps.values()))

# a full target should have zero gap once achieved
stats2 = m._new_committed_stats(cfg)
big = dict(stats["counts"])
for k in big:
    big[k] = 100000
stats2["counts"] = big
for name in stats["hists"]:
    stats2["hists"][name] = [100000] * len(stats2["hists"][name])
check("no gaps when saturated", m._evaluate_gaps(stats2, targets) == {})

# ── task scoring (fake blueprint task with a summary) ────────────────
class FakeHist(object):
    def __init__(self, counts):
        self.counts = counts
        self.edges = list(range(len(counts) + 1))

    def total(self):
        return sum(self.counts)


class FakeSummary(object):
    def __init__(self):
        self.macro_tick_total = 60
        self.macro_pass_count = 40
        self.macro_normal_count = 10
        self.macro_turn_left_count = 5
        self.macro_turn_right_count = 5
        self.local_direct_count = 20
        self.local_avoidance_count = 40
        self.macro_correction_angle_hist = FakeHist(
            [0, 0, 0, 0, 0, 0, 0, 10, 0, 0])
        self.macro_correction_distance_hist = FakeHist([0, 0, 10, 0, 0])
        self.local_deflection_hist = FakeHist(
            [0, 0, 10, 10, 5, 0, 0])
        self.local_yaw_rate_hist = FakeHist([0, 0, 0, 0, 40, 0, 0, 0, 0])
        self.local_speed_hist = FakeHist([5, 5, 5, 5, 5, 5])


class FakeTask(object):
    def __init__(self, task_id):
        self.task_id = task_id
        self.summary = FakeSummary()


t = FakeTask(7)
check("task macro:pass contribution",
      m._task_contribution(t, "count:macro:pass") == 40.0)
check("task macro:turn_left contribution",
      m._task_contribution(t, "count:macro:turn_left") == 5.0)
check("task corr_angle bin7 contribution",
      m._task_contribution(t, "hist_bin:macro_correction_angle:7") == 10.0)
check("task deflection bin2 contribution",
      m._task_contribution(t, "hist_bin:local_deflection:2") == 10.0)
check("task corr_angle total contribution",
      m._task_contribution(t, "hist_total:macro_correction_angle") == 10.0)

# A task that contributes to a gap should score higher than one that
# contributes nothing to the current gaps.
gaps = m._evaluate_gaps(stats, targets)
score_t = m._score_task_for_gaps(t, stats, targets)
class EmptyTask(FakeTask):
    pass


e = EmptyTask(8)
e.summary = FakeSummary()
for name in ["macro_correction_angle_hist", "macro_correction_distance_hist",
             "local_deflection_hist", "local_yaw_rate_hist",
             "local_speed_hist"]:
    e.summary.__dict__[name] = FakeHist([0] * len(getattr(t.summary, name).counts))
e.summary.macro_pass_count = 0
e.summary.macro_normal_count = 0
e.summary.macro_turn_left_count = 0
e.summary.macro_turn_right_count = 0
e.summary.local_direct_count = 0
e.summary.local_avoidance_count = 0
e.summary.macro_tick_total = 0
score_e = m._score_task_for_gaps(e, stats, targets)
check("gap-contributing task scores higher",
      score_t > score_e + 1.0)

# top-up config default
check("max_rounds default",
      int((m._g.get("blueprint_generation", {}).get("topup", {}) or {}).get(
          "max_rounds", 3)) == 3)

print("\nALL %d CHECKS PASSED" % passed)
