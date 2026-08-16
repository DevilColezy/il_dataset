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

## Scene generation (no silent degradation)

`SceneTaskBlueprintGenerator` (C++):

- **Explicit strata schedule of 10 scenes**: scene 0 = explicit EMPTY /
  CLEAR scene (0 obstacles, radius class `none`); scenes 1..9 = non-empty
  `sparse/medium/dense` × `small/medium/large` with count/radius bands
  ALIGNED to the configured classification thresholds
  (`density_sparse_max`/`density_dense_min`/`radius_small_max`/
  `radius_large_min`) so planned == actual class on success.
  `scene_count < 10` never claims full coverage.
- **Coverage reflects the ACTUAL geometry, never `scene_id`**: a stratum
  is covered only when the REAL obstacle set classifies into the planned
  density AND radius class.  An empty scene is never counted as `small`
  radius and its tasks never contribute to the radius quotas.
- **Item 八**: the requested obstacle count is sampled ONCE from the scene
  base seed; every whole-scene retry keeps the SAME count / count stratum /
  radius stratum and only re-samples positions / radii (retries never
  lower the task difficulty).  `actual != requested` ⇒ `generation_valid
  = false`.
- Every scene records `requested_obstacle_count` and
  `actual_obstacle_count`, `generation_valid` and `failure_reason`.
  A whole scene is **retried with a fresh placement seed** up to
  `max_generation_attempts`; a still-failed scene fails generation
  explicitly (never a silently degraded scene).
- Every cylinder is verified to keep the minimum surface gap and the
  boundary margin (analytic).
- **ESDF semantics (documented in the header)**: `dist[id]` = drone
  CENTRE → obstacle SURFACE distance.  A free cell (start/goal/connectivity)
  requires `dist > free_cell_surface_clearance_m` (must be ≥
  `vehicle.radius_m`, validated) AND boundary clearance.  Body-edge
  clearance of a free cell = `free_cell_surface_clearance_m − vehicle_radius`.
- `labelComponents` flood-fill is 8-connected with **no diagonal
  corner-cutting**: a diagonal move is allowed only when BOTH orthogonal
  neighbours are free.
- Start/goal are picked from free-cell CENTRES of the **same connected
  component** with distance in the task stratum, then analytically
  re-checked by `verifyEndpointPair` (surface clearance, boundary margin,
  distance band).

## Task generation (oversampling + hard quotas)

- **candidate_pool_multiplier is actually used** (item 四): each scene
  samples + preflights until its candidate pool reaches
  `tasks_per_scene * max(1, candidate_pool_multiplier)` or the finite
  `qualification_attempt_budget` is exhausted.  A budget shortfall below
  the candidate target is an EXPLICIT `generation_ok=false` failure
  (`pool_budget_exhausted`, reason includes pool vs target) — it is never
  reported as "oversampled".  The manifest records the three counts
  separately: `tasks_sampled` (candidates), `tasks_pool_accepted`
  (preflight-accepted pool) vs `tasks_pool_target`, and
  `tasks_quota_accepted` (final quota-selected tasks).
- `applyQuotas` (item 五) first establishes a per-scene floor for EVERY
  scene_id 0..scene_count-1 (a scene absent from the pool counts as 0),
  then the global required-behavior + TURN side quotas, then the
  density/radius/distance level floors, then fills to capacity
  (per-scene cap `tasks_per_scene`, total cap scene_count×tasks_per_scene).
- **TURN left/right quotas are real TURN statistics (item 六)**: only
  tasks that ACTUALLY issued TURN_LEFT / TURN_RIGHT directives contribute
  to `min_turn_per_side` / left-right balance.  A NORMAL_CORRECTION
  direction-token laterality is never a TURN.  `BlueprintTask` records
  `saw_turn_left`/`saw_turn_right`/`saw_normal_correction`/`turn_update_count`/
  `normal_update_count`.  Phase B `need_behavior` applies ONLY to the
  hard-required `kRequiredBehaviors` — turn_both / multi_correction /
  long_takeover never fill capacity ahead of the required classes.
- Any unmet quota is recorded in `BlueprintResult.unmet_quotas` and sets
  `generation_ok = false`.
- `BlueprintResult` carries `generation_ok`, `failure_reason`,
  `unmet_quotas`, `requested_scenes`, `requested_tasks_per_scene`,
  `scenes_valid`, `strata_required/covered/flags`,
  `per_scene_accepted` (fixed length scene_count),
  `category_counts`, `tasks_sampled`/`tasks_preflighted`/
  `tasks_pool_target`/`tasks_pool_accepted`/`tasks_quota_accepted`,
  `pool_budget_exhausted`; each `BlueprintScene` carries `stratum_id`,
  planned/actual density+radius classes, actual min/max radius,
  requested/actual obstacle counts, `generation_valid`, `failure_reason`;
  each `BlueprintTaskAudit` carries `out_of_bounds`,
  `qualification_exceeded`.
- **Gate (item 十一)**: `il_manager.py` always writes the manifest, then
  checks `generation_ok` BEFORE any blueprint_only/dry_run early return:
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
