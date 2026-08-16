# il_dataset — hierarchical local expert collection (single production path)

## Overview

`il_dataset` collects Flightmare/AvoidBench imitation-learning datasets.
The **only** production path uses **one** C++ expert — the stable
"5 Hz local target corrector + 30 Hz local obstacle-avoidance expert"
(`HierarchicalExpert`, namespace `il_dataset::expert`, library
`il_hierarchical_expert_lib`, pybind module `_il_hierarchical_expert`),
ported from the read-only reference `il_2d_multiscale_debug`.

- **One node** (`scripts/il_manager.py`) is started by BOTH launches
  (`il_dataset_collect.launch` and `il_dataset_joint_v2_collect.launch`);
  there is no legacy/alternative expert path and no config-based dispatch.
- The manager instantiates **exactly one** `HierarchicalExpert`; it is the
  **sole generator of every horizontal control command** recorded and sent
  to Flightmare.
- All legacy expert code (`CausalLocalTargetStream`,
  `PrivilegedMicroDetourPlanner`, Python `il_macro_expert.py` /
  `il_micro_expert.py`, old goal switcher, old preflight/task classifier,
  old generator chain, `debug_viewer.py`) and the legacy test directory
  have been **deleted**.  The C++ build never builds them, even under
  `CATKIN_ENABLE_TESTING`.  There is no lossy `expert_mode` projection:
  the dataset carries `hierarchical_mode` only, and `src/save_net` reads
  it directly.

## Architecture

```
Flightmare depth (Unity) ──┐
Flightmare state (flightlib)┼──► Python (thin: lifecycle / sync / write)
                            │        │  step(pos, yaw, vel, yaw_rate, depth,
                            │        │       cam_pose, tick, collision)
                            ▼        ▼
                 _il_hierarchical_expert (C++17, ONE state-owning instance)
                 ┌──────────────────────────────────────────────────────┐
                 │ Flightmare2DObservation : depth → current FOV patch  │
                 │ ObservedGrid2D         : causal local history        │
                 │ VisibilityTargetCorrector (5 Hz, tick%6==0)          │
                 │ EffectiveTargetAdapter (every 30 Hz tick)            │
                 │ LocalPlanner30Hz        (every 30 Hz tick)           │
                 │ HierarchicalExpertFsm   (states / terminal)          │
                 └──────────────────────────────────────────────────────┘
                          │  ExpertStepOutput (flat, all CSV fields)
                          ▼
                 Python: altitude-hold vz merged → Flightmare command
                          + DatasetWriter (schema v25 + 5 Hz extension)
```

Flightmare dynamics (`src/flightmare_dynamics/pybind_module.cpp` →
`_flightmare_dynamics`) is a standalone pybind module that wraps the
flightlib quadrotor backend; it is the only Flightmare control interface.

## Camera projection (single extrinsic formula)

`CameraRig2D` (in `flightmare_2d_observation.hpp/.cpp`) is the ONE place
that converts depth pixels into world rays.  The same camera semantics are
used by the runtime observer and by the `PreflightSimulator`:

- Flightmare world: x-fwd, y-left, z-up.  Expert 2D frame: X-right,
  Y-up.  `expert_yaw = flightmare_yaw + π/2`.
- Unity `T_BC` (camera→Unity body, 16 floats row-major) is converted to
  flightlib body: `t_fl = [tx, tz, ty]`,
  `R_bc_fl = P · R_unity · C` with permutation `P:[x,y,z]→[x,z,y]` and the
  optical flip `C = diag(1,-1,1)`.
- **Single application** of the translation:
  `cam_world = vehicle_world + R_WB · t_fl`
  `point_world = cam_world + R_WB · (R_bc_fl · point_optical)`
- Camera world yaw (FOV centre) = XY heading of `R_WB · (R_bc_fl · [0,0,1])`.
- Ray direction for bearing θ: `R_WB · (R_bc_fl · rotZ(θ) · [0,0,1])`,
  projected to XY.

The `depth.t_bc` config entry is the SINGLE source of the camera→body
matrix: `il_common.make_depth_vehicle` reads the SAME validated 16-element
`depth.t_bc` (no second hard-coded matrix), `il_expert_config` feeds it into
`Params2D.cam_t_bc_*`/`cam_r_bc` for the C++ `CameraRig2D`, and
`il_config.load_config()` injects the UNIQUE default when a config omits it.
`il_config.py` validates the matrix (16 finite floats, last row
`[0,0,0,1]`, orthonormal 3×3 with det +1), that `depth.fov ==
hierarchical_expert.observation.fov_deg` and that `depth.max_m >=
observation.range_m`.  The writer metadata records the actual `depth.t_bc`.

## Blueprint generation (deficit-driven, offline, C++17)

`SceneTaskBlueprintGenerator` is now a thin facade over
`BlueprintGenerationController` (the old fixed strata schedule + hard
`applyQuotas` path was REMOVED — there is exactly ONE selection path).

**Warehouse (single source)**: the only free region is
`blueprint_generation.warehouse.free_region` = `[-7,10] x [0,30]` (must
equal `hierarchical_expert.region`, validated).  The outer
`wall_extension_m` (1 m) is the non-traversable envelope.  No fake
internal static structure is ever constructed.  Out-of-bounds ALWAYS
means crossing the FREE region boundary (matching the real Flightmare
truth audit); the wall shell is only optionally visible in the synthetic
depth observation (`walls_visible_in_observation`).

**Scene Profiles (scene generation)**: `SceneProfileGenerator` realizes
random scenes from a profile catalog — `empty`, `sparse_tiny`,
`dense_tiny` (20-30 cylinders of r≈0.1 m), `sparse_small`, `dense_small`,
`sparse_medium`, `dense_medium`, `large_single` (r up to ≈6 m),
`large_sparse`, `mixed_tiny_small`, `mixed_small_medium`,
`mixed_small_large`, `mixed_all`, `clustered`, `corridor`, `bottleneck`,
`chicane`, `central_blocker`, `edge_clutter`.  Every placement keeps:
  * boundary margin inside the free region;
  * pairwise surface gap ≥ `min_surface_gap_m`, which is validated to be
    ≥ `plannerRequiredPassage() = 2·(vehicle_radius + navigation_clearance
    + discretisation_margin) + 2·generation_margin` — a passage that
    satisfies the physical drone radius but NOT the local planner's
    clearance / grid discretisation is never generated;
  * structured profiles (corridor / bottleneck / chicane) keep a central
    passage of width ≥ planner-required passage.
A whole scene is retried with a fresh placement seed up to
`max_scene_generation_attempts`; a still-failed scene is recorded with
`generation_valid=false` and skipped (never silently degraded).  Each
`BlueprintScene` carries `profile` + `metadata` (radius bands, density
proxy, cluster count, free-space ratio, corridor-width proxy,
geometry/planning validity) plus the legacy strata fields (report-only).

**Geometry cache (per scene, built ONCE)**: `SceneGeometryCache` builds
the truth ESDF over the free region, 8-connected components (no diagonal
corner-cutting), the main-component area and the list of VALID task cells
— reused by every task candidate of the scene (never rebuilt per task).

**Task candidates**: `TaskCandidateGenerator` samples start/goal from the
cached valid cells, classifies a cheap geometric PROXY (CLEAR /
LOCAL_AVOIDANCE / OFFSET_AVOIDANCE / LARGE_OCCLUSION / MULTI_OBSTACLE /
CHICANE / NARROW_BUT_PLANNABLE / LONG_DETOUR) to bias the goal, and
samples the INITIAL YAW from a layered distribution over the absolute
goal-bearing error (`0-15 / 15-35 / 35-55 / 55-90 / 90-150 / 150-180 deg`,
weights emphasise 35-55 — the ±45° FOV decision boundary — while the wide
bins keep out-of-FOV / rear-goal turns), with mirror-balanced signs.
The initial yaw is no longer limited to goal-heading ±15°.  Classification
is O(1) per pair (narrow passages / large obstacles are cached once per
scene), and LONG_DETOUR / NARROW_BUT_PLANNABLE use DIRECTIONAL sampling
(start/goal on opposite sides of a blocker / passage centre) instead of
waiting for a lucky random pair.

**Preflight + distribution summary**: every candidate first passes a
CHEAP staged filter (bounds / clearance / distance / main component /
straight swept segment) and only then runs the closed-loop
`PreflightSimulator` (the SAME expert).  The synthetic observation is an
ANALYTIC ray cast (closed-form ray-circle + ray-rectangle slab, no
spatial marching); when `synthetic_observation.walls_visible_in_observation`
is true the warehouse wall envelope appears in the synthetic depth, but
it NEVER changes the out-of-bounds audit (that stays the FREE region
`[-7,10] x [0,30]`, matching the real Flightmare truth audit).  Each
preflight produces a `TaskDistributionSummary`: actual path length +
stretch ratio, 5 Hz tick-level PASS/NORMAL/TURN_LEFT/TURN_RIGHT counts +
correction-angle and correction-distance histograms, 30 Hz deflection /
yaw-rate / speed histograms (deflection skipped below
`min_deflection_speed_mps` to avoid NaN), min/mean observed clearance, and
a 2D synthetic raycast DEPTH PROXY (near/mid/far/free counts at a
temporal stride).  Preflights that stall or make no progress are cut
short by the `early_termination` detector (blueprint-only, never changes
expert labels) so the budgets are spent on viable candidates.

**Distribution targets + deficits (quotas replaced)**: the global
`DistributionAnalyzer` accumulates the summaries and computes per-target
deficits (5 Hz coverage, 30 Hz deflection, depth bands, yaw bins, path
length classes, left/right balance).  Missing types steer the next
generation round (weighted profile / task-type / yaw-stratum sampling);
soft shortfalls only produce warnings.  `evaluateCoverage` adds GROUPED
HARD checks (deflection strong-right/right/near-direct/left/strong-left
and correction right/near/left groups — an anti-degeneracy guard against
a pool that passes per-bin soft targets while piling into one side).  The
final `select()` is a deterministic greedy scorer (contribution to
deficient targets minus over-supply penalties) with a balance-aware
marginal bonus for the minority turn / yaw side, and a TWO-STAGE flow:
greedy coverage first, then a scene-consolidation pass that drops entire
scenes whose tasks are not needed for hard coverage (fewer scenes, more
complementary tasks per scene).  `selection_score` is written per task.

**Budgets**: `max_scene_candidates`, `max_task_candidates_per_scene`,
`max_generation_rounds`, `max_total_preflight_tasks`,
`max_total_preflight_ticks`, `max_preflight_ticks_per_task`,
`max_scene_generation_attempts`, `max_task_generation_attempts`.
The preflight budgets are enforced on ATTEMPTS and TICKS (success +
failure) — never on the accepted pool size alone.  Reaching a budget ends
the run normally with the achieved distribution + remaining deficits and
a `budget_exhausted_reason`.  The result reports efficiency diagnostics:
`preflight_attempt_count / preflight_success_count / preflight_failure_count`,
`total_preflight_ticks`, `full_preflight_attempted/success`,
`preflight_acceptance_ratio`, `selected_per_preflight_ratio`, and per-round
`round_logs` (also logged to stderr with `early_termination.log_rounds`).

**generation_ok (new semantics)**: false ONLY when a HARD minimum
coverage / structural balance / scene / task-count gate fails; soft
target shortfalls are reported in `remaining_deficits` / `warnings` with
`generation_ok = true`.

**Gate**: `il_manager.py` always writes the manifest, then checks
`generation_ok` BEFORE any blueprint_only/dry_run early return:
`generation_ok == false` ⇒ `logerr` + `RuntimeError` in ALL modes
(normal mode never connects Flightmare; no mode prints a success line);
only after `generation_ok == true` do blueprint_only / dry_run end
normally.

## Preflight continuous audit

`PreflightSimulator::step()` audits **every** simulated step
continuously: point collision at the current state, integration, then
swept collision of the segment prev→new, plus the new point and the
boundary crossing.  The boundary-sweep check uses the ONE shared helper
`segmentDiskInsideBounds` in `types.hpp` (item 七): the region rectangle is
convex, so a straight segment swept by the drone disk stays inside IFF both
endpoints lie in the r-shrunk rectangle — exact, and used identically by
`TruthCylinderAudit` and `PreflightSimulator` (the old endpoint-only
`segSegDist` approximation is removed).  Coverage includes the initial
point, every executed segment's start AND end, and the state after the
final terminal command (the last segment's new point).  The
synthetic patch is built by ray-marching from the **camera** (not the
drone centre) against the exact cylinder truth using the SAME
`CameraRig2D`/`buildFromRays` code path as runtime.  A task is accepted
only when the expert reaches the goal with no point/swept collision, no
boundary crossing, valid macro labels, within budget.

## Runtime strict 30 Hz frame state machine

`il_manager.py` `_episode` guarantees a contiguous 30 Hz committed
sequence:

- `control_tick` (data.csv `episode_frame_index`) advances **only** when a
  render frame matches the exact requested `frame_id` (per-attempt wait
  `sync.frame_match_timeout_s`), the expert step succeeds, and the command
  is actually executed.  Failed attempts retry on the **same saved state**
  with a fresh render frame id up to `commit.max_frame_retries`; they never
  advance dynamics, never advance the expert, and never write a row.
  Exceeding the retries stops and **rejects** the episode
  (`frame_retries_exceeded`).
- `macro_update_mask` is set on real 5 Hz rows (every 6 committed ticks).
- `sync.csv` records render attempts (including dropped/mismatched
  attempts) separately from `data.csv` control ticks.
- `fsm.trajectory_wall_timeout_s == 0` disables the wall-clock flight
  timeout entirely (an episode ends only on an expert terminal state or
  shutdown).
- Every **executed** dynamics step is audited continuously (first and last
  segment included) with the exact truth cylinders: `segmentCollision` on
  pos→pos_after, boundary crossing, per-obstacle brake risk
  (`truth_brake_risk` = MAX over obstacles of closing-speed stopping
  distance + envelope), plus Unity's own collision flag.  An episode is
  committed only if it reaches the goal with none of
  collision/out-of-bounds/brake/label/latency/rejection flags set.

## Two-level expert (hierarchical_local_v1)

- **5 Hz VisibilityTargetCorrector** answers "is the current FOV enough
  for the 30 Hz expert to finish its own avoidance?" and outputs a
  zero-order-held `TargetCorrectionDirective`:
  `PASS_THROUGH` / `NORMAL_CORRECTION` / `TURN_LEFT` / `TURN_RIGHT`
  (one bounded world-latched rotation step).  It reads ONLY current patch /
  causal history / vehicle state / original goal / its own memory — never
  the 30 Hz outcome, truth ESDF, obstacle truth or a global path.
- **EffectiveTargetAdapter** (every 30 Hz tick) converts the directive at
  the live pose into the 30 Hz information bottleneck (body-FLU unit
  direction + normalized distance, and the world `LocalTarget` for the
  planner).
- **LocalPlanner30Hz** samples speed×lateral×yaw-rate candidates, rolls
  them out under the shared kinematics, validates against current FOV +
  known-FREE + soft/dynamic clearance, truncates to the executable safe
  prefix, and scores by progress/clearance/smoothness/alignment/risk.

## Target encoding protocol (R = 5.0 m, reserve = 0.5 m)

- `direction_bin_count = 11`; token 0 = `TURN_LEFT`; tokens 1..11 =
  ordinary in-FOV bins left→right (0° included); token 12 = `TURN_RIGHT`.
- Ordinary target: `distance_norm = min(real_dist, R - 0.5) / R`
  (max = 0.9, strictly `< 1`).
- TURN: `distance_norm == 1.0` exactly → pure rotation, horizontal
  translation 0; the world direction is latched and re-expressed in the
  live body frame every 30 Hz tick (finite turns, no infinite spin).
- Goal reached: `distance_norm == 0`, canonical direction `(1,0,0)`.

## Fields (data.csv)

### 30 Hz student inputs (schema v25 semantics, NEW meaning)
`depth_file`, `gravity_flu_*`, `velocity_flu_*`, `yaw_rate_flu`,
`goal_direction_flu_*`, `goal_distance_clipped_m`, `goal_distance_norm`.

`goal_*` encode the **effective target** of the hierarchical expert:
- PASS → original goal;
- NORMAL → live re-expressed world-latched correction;
- TURN → live re-expressed world-latched turn direction, norm = 1.

### 30 Hz supervision
`target_velocity_flu_*`, `target_yaw_rate` — the FINAL command actually
sent to the Flightmare backend (after speed/accel/yaw-rate/yaw-accel
limits; NOT an intent, NOT a trajectory-end velocity).
`velocity_command_flu_*` / `yaw_rate_command` are identical aliases.

`hierarchical_mode` is the ONLY mode label (8 states):
`direct` / `local_avoidance` / `macro_normal` / `macro_turn_left` /
`macro_turn_right` / `turn_to_target` / `goal_capture` / `blocked`.
There is **no** legacy lossy `expert_mode` projection.  `planner_status`
carries the C++ `PlannerStatus` STRING (SAFE_PROGRESSING / TURNING /
EMERGENCY_BRAKE / ...); the save_net loader maps the string to a stable
index and raises on unknown values (never a silent -1).

### 5 Hz supervision (new, `two_level_expert_labels_v1`)
`macro_update_mask` (1 on real 5 Hz frames, every 6 committed ticks),
`macro_label_valid`, `macro_correction_type`, `macro_direction_token`,
`macro_direction_flu_*`, `macro_distance_norm`, `macro_param_valid`.

Rules: train the 5 Hz student ONLY on `macro_update_mask==1` rows; never
repeat one 5 Hz decision 6 times; an update frame without a legal label
rejects the whole episode; `macro_direction_flu_*` is the unit direction
decoded at the 5 Hz decision instant (z=0); `macro_param_valid==1` for
NORMAL/TURN, 0 for PASS.  The save_net loader (item 三) STATICALLY audits
committed episodes at discovery (mask grid, label validity, per-type
PASS/NORMAL/TURN_LEFT/TURN_RIGHT token+param+distance rules, unit-vector
and finiteness) and mirrors 5 Hz fields exactly: only the LEFT axis of the
goal direction / macro direction mirrors, TURN_LEFT↔TURN_RIGHT (type +
`hierarchical_mode`) swap, tokens mirror 0↔12, 1↔11, … (PASS -1 stays),
never x/z.

### 5 Hz student inputs (new)
`navigation_goal_direction_flu_*`, `navigation_goal_distance_clipped_m`,
`navigation_goal_distance_norm` — the ORIGINAL navigation goal (never the
effective `goal_*`).  The 5 Hz student also reuses the current depth /
gravity / velocity / yaw-rate from the 30 Hz row.

> Training scope (item 十二): `src/save_net/train.py` currently trains the
> 30 Hz local-avoidance student ONLY.  The 5 Hz labels are aligned,
> validated and exposed by the loader (`state_5hz` / `label_5hz`) for an
> independent 5 Hz trainer; no 5 Hz network is implemented or claimed yet.

### Privileged diagnostics (never student inputs)
`effective_target_world_*`, `original_navigation_goal_world_*`,
`correction_{enter,exit,update}_event`, `observability_*`,
`directive_update_event`, `mission_revision`, `fsm_state`,
`planner_failure_reason`, truth audit fields
(`truth_minimum_clearance_m`, `truth_brake_risk`, `observed_brake_risk`,
`truth_brake_would_trigger`), etc.

## Metadata

- `schema_version` stays **25** (save_net compatibility).
- `schema_extensions: ["two_level_expert_labels_v1"]`
- `expert_stack_revision: "hierarchical_local_v1"`
- `hierarchical_modes`, `student_input_fields_30hz`,
  `supervision_fields_30hz`, `student_input_fields_5hz`,
  `supervision_fields_5hz`, `macro_update_hz`, `local_control_hz`,
  `target_encoding_contract`, `information_boundary_contract`,
  `compatibility_aliases`, `sequence_contract`.
- No `expert_mode_compat_map` / `expert_mode_compat_contract` (the lossy
  projection is gone).

## Build & run

```bash
catkin build il_dataset
# Both launches run the SAME production node; the default config for both
# is the joint_v2 recipe:
roslaunch il_dataset il_dataset_collect.launch
roslaunch il_dataset il_dataset_joint_v2_collect.launch

# Custom config / dry-run / blueprint-only:
roslaunch il_dataset il_dataset_collect.launch \
    config_file:=/path/to/my_config.yaml dry_run:=true blueprint_only:=true
```

The launch never starts or stops AvoidBench: start it separately first
(see `launch/il_dataset_collect.launch` header).

## Single source of truth

Every expert parameter lives in `global.hierarchical_expert` (YAML) →
`scripts/il_expert_config.py` → C++ `Params2D`.  No second defaults exist
in C++ or Python.  The perception range is `R = 5.0 m` everywhere (never
hard-coded 5/6 in a module).  `REQUIRED_MODULES` (see
`scripts/il_config.py`) lists only new-architecture modules.

## Note on `src/save_net/rollout.py`

`save_net/rollout.py` (and its test) is **legacy deployment tooling** that
predates this pipeline: it is standalone (does not import
`save_net/dataloader.py`) and references removed fields (`goal_switch_event`,
`active_goal_*`, `abrupt_goal_switch`, legacy modes).  It is left intact as
a deployment artifact and is NOT part of the new collection pipeline or the
new schema; it must not be used to train/deploy against v25
`hierarchical_mode` data.
