#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <string>
#include <vector>

namespace il_dataset {

/// Vehicle state in ROS world coordinates (x-fwd, y-left, z-up).
/// Yaw follows convention B: yaw=0 -> body +Y faces world +Y,
/// yaw = atan2(world_y, world_x) - pi/2.
struct VehicleState {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    double yaw = 0.0;
    double yaw_rate = 0.0;
};

/// A single point on a dense time-sampled trajectory.
struct TrajectoryPoint {
    double t = 0.0;
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    double yaw = 0.0;
    double yaw_rate = 0.0;
    /// ESDF clearance at this point (negative = collision, after drone
    /// radius subtraction). NaN when unknown / outside the map.
    double clearance = 0.0;
};

/// Status codes returned by the local trajectory planner.
enum class PlannerStatus : int {
    SUCCESS = 0,
    INVALID_INPUT = 1,
    NO_SAFE_MOTION = 2,
    LOCAL_TERMINAL_INVALID = 3,
    OPTIMIZATION_FAILED = 4,
    COLLISION = 5,
    DYNAMICS_VIOLATION = 6,
    UNKNOWN_SPACE = 7,   // trajectory enters unknown space (not just low clearance)
    SEARCH_FAILED = 8,   // A*/JPS found no usable local path
    EMERGENCY_HOLD = 9,
};

/// 5 Hz macro navigation state machine (section IV).
enum class MacroMode : int {
    DIRECT_GUIDE = 0,
    SIDE_GUIDE = 1,
    OBSERVE = 2,
    GOAL_REACHED = 3,
    FAILED = 4,
};

/// Committed horizontal bypass side. FLU: +y is left.
enum class Side : int {
    NONE = 0,
    LEFT = 1,
    RIGHT = -1,
};

/// Low-frequency query result from the 30 Hz local planner to the 5 Hz
/// macro expert: whether the direct goal intent can be recovered by the
/// local system within the finite horizon and current known space
/// (section V / VI).
enum class RecoverabilityStatus : int {
    /// Local system really can resolve the direct intent (full known-free
    /// path within horizon, terminal re-aligns with the guide direction).
    DIRECT_REJOIN_SUCCESS = 0,
    /// Some progress was found but the rejoin point is not reachable.
    PARTIAL_PROGRESS_ONLY = 1,
    /// Direct intent is blocked by a known occupied region.
    BLOCKED_BY_KNOWN = 2,
    /// Direct intent is blocked because the corridor is still unknown.
    BLOCKED_BY_UNKNOWN = 3,
    /// No safe motion exists from the current state.
    NO_SAFE_MOTION = 4,
};

/// Result of the local recoverability query (section VI).
struct RecoverabilityResult {
    RecoverabilityStatus status = RecoverabilityStatus::NO_SAFE_MOTION;
    bool feasible = false;
    bool known_free = false;
    double minimum_clearance = 0.0;
    double estimated_duration = 0.0;
    double goal_progress = 0.0;
    /// Terminal TANGENT alignment with the direct-guide direction (uses the
    /// path end tangent, not the start-end chord).
    double terminal_guide_alignment = 0.0;
    double path_length = 0.0;
    /// Actual rejoin target distance (min(configured, guide distance)).
    double rejoin_distance = 0.0;
    /// detour_ratio = path_length / straight_rejoin_distance.
    double detour_ratio = 0.0;
    Eigen::Vector3d rejoin_point{Eigen::Vector3d::Zero()};
    /// Stable world-geometry blocker signature (same as GoalBlocker).
    int blocker_signature = -1;
    bool left_edge_visible = false;
    bool right_edge_visible = false;
    bool left_corridor_known = false;
    bool right_corridor_known = false;
    std::string reason;
};

enum class CandidateType : int {
    DIRECT = 0,
    SIDE = 1,
    OBSERVE = 2,
    GOAL_FRONTIER = 3,
    PREVIOUS_CONTINUATION = 4,
};

/// A single macro navigation candidate (section VIII).
struct MacroCandidate {
    CandidateType type = CandidateType::DIRECT;
    Side side = Side::NONE;
    Eigen::Vector3d position_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d position_flu{Eigen::Vector3d::Zero()};
    /// True only when a real observed-map path search reached the candidate
    /// point (full goal reached).
    bool known_reachable = false;
    bool full_goal_reached = false;
    bool found_partial = false;
    /// Observed-map A* path cost (length) from the current state.
    double observed_path_cost = 0.0;
    double observed_path_length = 0.0;
    double minimum_clearance = 0.0;
    double goal_progress = 0.0;
    bool left_edge_visible = false;
    bool right_edge_visible = false;
    double unknown_information_gain = 0.0;
    // Privileged fields (filled only by the PrivilegedOracle).
    bool connected_to_goal = false;
    double global_cost_to_go = 0.0;
    double global_clearance = 0.0;
    double global_path_length = 0.0;
    double privileged_score = 0.0;
    std::string source;
};

/// Result of evaluating one start-goal task candidate on the built scene
/// map (task generation; sections XXVIII/XXXIX).  All fields are pure
/// privileged diagnostics — never student inputs and never fed to the
/// macro expert as a mode hint.
struct TaskCandidateResult {
    bool start_free = false;
    bool goal_free = false;
    bool goal_reachable = false;
    double straight_distance = 0.0;
    /// Global shortest-path length from start to goal (m).
    double global_path_length = -1.0;
    /// global_path_length / max(0.1, straight_distance).
    double global_detour_ratio = -1.0;
    /// Minimum ESDF clearance (vehicle radius already subtracted) along
    /// the reconstructed global path.
    double global_min_clearance = -1.0;
    bool direct_blocked = false;
    /// Number of distinct blocked runs along the direct start->goal ray.
    int direct_blocker_count = 0;
    /// Distance (m) from start to the first blocked cell on the direct ray.
    double nearest_blocker_distance_m = -1.0;
    /// Local-scale audit on the FULL map (short-range rejoin search with
    /// bypass allowed) — the same semantics as PrivilegedInterventionOracle.
    bool privileged_local_recoverable = false;
    bool left_global_feasible = false;
    bool right_global_feasible = false;
    double left_path_length = -1.0;
    double right_path_length = -1.0;
    std::string reason;
};

/// Configuration for batch task-candidate evaluation (sections
/// XXVIII/XXXIX/LIX).  The local-audit bounds mirror local_recoverability
/// / privileged_intervention so generated classes match the real scale
/// definition (behaviour scale, not obstacle size).
struct TaskGenerationConfig {
    /// Extra clearance for start/goal free tests (unified ESDF semantics:
    /// vehicle radius already subtracted, so this is an additional margin).
    double start_clearance_m = 0.45;
    double goal_clearance_m = 0.45;
    /// Extra clearance for the direct-corridor ray walk.
    double direct_corridor_clearance_m = 0.45;
    double min_task_distance_m = 3.0;
    double max_task_distance_m = 30.0;
    /// Lateral probe geometry for left/right global feasibility.
    double lateral_probe_offset_m = 1.2;
    double lateral_probe_spacing_m = 0.6;
    int lateral_probe_count = 4;
    double lateral_path_clearance_m = 0.45;
    // ── Local-scale audit capability bounds (same as local_recoverability).
    double search_clearance_m = 0.25;
    double search_max_time_ms = 20.0;
    double rejoin_distance_m = 2.5;
    double max_duration_s = 2.5;
    double max_path_length_m = 6.0;
    double nominal_speed_mps = 1.8;
    double max_detour_ratio = 1.6;
    double min_goal_progress_m = 0.30;
    double min_terminal_alignment = 0.5;
    double terminal_tangent_min_baseline = 0.3;
    double search_lateral_margin_m = 2.0;
    double search_longitudinal_margin_m = 2.0;
};

/// Identified goal blocker in the observed map (section IV.2).
struct GoalBlocker {
    bool found = false;
    /// Stable world-geometry signature (quantized centroid + extent +
    /// blocking-ray depth on a 0.5 m world grid).  It is NOT a map-native
    /// component index: the macro layer performs the cross-tick
    /// association using centroid distance + bbox overlap.
    int blocker_signature = -1;
    Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
    Eigen::Vector3d bbox_min_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d bbox_max_world{Eigen::Vector3d::Zero()};
    double extent = 0.0;
    /// Distance (m) from the vehicle to the first blocked cell along the
    /// goal ray.
    double blocking_ray_depth = -1.0;
    /// Number of cells in the blocked connected component.
    int component_cell_count = 0;
    /// True when the blocking component is known occupied (vs unknown).
    bool blocked_by_known = false;
    Eigen::Vector3d left_edge_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d right_edge_world{Eigen::Vector3d::Zero()};
    bool left_edge_visible = false;
    bool right_edge_visible = false;
    bool left_corridor_known = false;
    bool right_corridor_known = false;
    Eigen::Vector3d left_corridor_point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d right_corridor_point{Eigen::Vector3d::Zero()};
};

/// The 5 Hz macro action, frozen in the world frame for the following
/// 200 ms (section IX).  Only the world-frame target and desired yaw are
/// authoritative; position_flu is regenerated each 30 Hz frame.
struct MacroAction {
    MacroMode mode = MacroMode::DIRECT_GUIDE;
    Side committed_side = Side::NONE;
    Eigen::Vector3d guide_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d guide_flu{Eigen::Vector3d::Zero()};
    double desired_yaw_world = 0.0;
    bool has_desired_yaw = false;
    double confidence = 1.0;
    double guide_distance = 0.0;
    bool is_new_tick = false;
    /// OBSERVE behaviour subtype (section XV): 0 = OBSERVE_ROTATE (pure
    /// rotation, zero translation), 1 = OBSERVE_MOVE (planned observation
    /// viewpoint reached through the normal 30 Hz pipeline).
    int observe_subtype = 0;
    /// OBSERVE direction metadata (LEFT / RIGHT / NONE).  Kept SEPARATE
    /// from committed_side: it is observation intent only and must never
    /// bias the local path search (section XIV).
    Side observe_side = Side::NONE;
    std::string reason;
};

/// 30 Hz local execution lifecycle mode (section XIII).
enum class ExecutionMode : int {
    TRACK_FRESH = 0,
    TRACK_CACHED = 1,
    ROTATE_ONLY = 2,
    BRAKE_HOLD = 3,
    EMERGENCY_STOP = 4,
};

/// 30 Hz local planning request (section XVII).
struct LocalPlanRequest {
    VehicleState state;
    /// Fixed world-frame macro guide from the held 5 Hz action.
    Eigen::Vector3d macro_guide_world{Eigen::Vector3d::Zero()};
    bool has_macro_yaw = false;
    double macro_yaw_world = 0.0;
    /// The final task goal (used to decide stop-at-goal and terminal
    /// selection only). Never used to synthesize hidden waypoints.
    Eigen::Vector3d goal_world{Eigen::Vector3d::Zero()};
    /// Warm-start remainder of the previous executed trajectory.
    std::vector<TrajectoryPoint> previous_trajectory;
    double previous_trajectory_age_s = 0.0;
    /// Committed macro side (SIDE_GUIDE): the search must respect it.
    Side committed_side = Side::NONE;
    /// forbid_unknown_space is always true for the local expert; kept to
    /// make the semantics explicit at every call site.
    bool forbid_unknown_space = true;
};

/// Result of a single 30 Hz local-planning invocation.
struct LocalPlanResult {
    bool success = false;
    PlannerStatus status = PlannerStatus::NO_SAFE_MOTION;
    std::string message;
    std::vector<TrajectoryPoint> trajectory;
    /// The macro guide waypoint (echoed, never the trajectory terminal).
    Eigen::Vector3d guide_waypoint{Eigen::Vector3d::Zero()};
    /// The actual trajectory terminal (local terminal, may be nearer than
    /// the macro guide).
    Eigen::Vector3d trajectory_terminal{Eigen::Vector3d::Zero()};
    double min_clearance = 0.0;
    double duration_s = 0.0;
    double planning_time_ms = 0.0;
    uint64_t plan_id = 0;
    int search_status = 0;  // 0 = full path, 1 = partial, 2 = no path
};

/// Final 30 Hz controller command (section XII).
struct ControllerCommand {
    Eigen::Vector3d velocity_flu{Eigen::Vector3d::Zero()};
    double yaw_rate = 0.0;
    bool valid = false;
};

/// Validation result for a trajectory.
struct ValidationResult {
    bool all_clear = false;
    bool any_collision = false;
    bool any_unknown = false;
    double min_clearance = 0.0;
    int clearance_violation_count = 0;
    Eigen::Vector3d worst_position{Eigen::Vector3d::Zero()};
    double worst_time = 0.0;
    double worst_clearance = 0.0;
};

/// Result of the swept-volume braking-risk check (section XVIII).
struct BrakeRiskResult {
    /// True when a collision risk exists along the predicted braking path.
    bool risk = false;
    double min_clearance = 0.0;
    /// Time (s) of the first risk along the braking trajectory, or -1.
    double first_risk_time = -1.0;
    /// Predicted braking distance (m).
    double braking_distance = 0.0;
};

/// Reason code of the privileged intervention evaluation (section III).
/// Enumerated — never free-form strings.  Describes WHY the direct intent
/// is NOT locally recoverable in the full map.
enum class InterventionReason : int {
    DIRECT_GLOBALLY_VALID = 0,      // privileged local recoverable
    DIRECT_LONG_WALL_BLOCKED = 1,   // local horizon cannot bypass (long wall / large area)
    DIRECT_GLOBAL_DISCONNECTED = 2, // guide region disconnected from the goal
    DIRECT_EXCESSIVE_DETOUR = 3,    // local bypass path too long / detours
    DIRECT_LOOP_RISK = 4,           // persistent loop in the trajectory history
    NO_GLOBAL_ROUTE = 5,            // current region has no global route to the goal
};

/// Precise failure category of the privileged LOCAL recoverability search
/// (section IX).  Only used for diagnostics / auxiliary labels.
enum class PrivilegedRecoverabilityFailure : int {
    NONE = 0,
    NO_REJOIN_PATH = 1,
    EXCESSIVE_PATH_LENGTH = 2,
    EXCESSIVE_DURATION = 3,
    EXCESSIVE_DETOUR = 4,
    LOW_CLEARANCE = 5,
    LOW_GOAL_PROGRESS = 6,
    BAD_TERMINAL_ALIGNMENT = 7,
};

/// Result of the privileged LOCAL-SCALE audit (section II/III).
///
/// The oracle answers the question: "given the full map, would the 30 Hz
/// local layer — with its own finite horizon and bypass ability — be able
/// to recover the direct intent (re-enter the direct guide)?"  It runs a
/// short-range search with geometry as consistent as possible with the
/// OBSERVED local recoverability, ALLOWING local bypass.  It never uses a
/// straight-line direct-ray collision check.
struct PrivilegedInterventionResult {
    /// Full map says the direct intent is locally recoverable.
    bool privileged_local_recoverable = true;
    bool privileged_rejoin_reached = false;
    double privileged_rejoin_distance = 0.0;
    double privileged_local_path_length = 0.0;
    double privileged_local_duration = 0.0;
    double privileged_detour_ratio = 0.0;
    double privileged_min_clearance = 0.0;
    double privileged_goal_progress = 0.0;
    /// Terminal TANGENT alignment with the direct-guide direction.
    double privileged_terminal_alignment = 0.0;
    /// Future macro intervention will likely be required (auxiliary label /
    /// episode analysis ONLY — never gates the main macro mode).
    bool privileged_future_intervention_required = false;
    PrivilegedRecoverabilityFailure failure_reason =
        PrivilegedRecoverabilityFailure::NONE;
    double current_cost_to_go = 0.0;
    double direct_cost_to_go = 0.0;
    bool loop_risk = false;
    InterventionReason reason = InterventionReason::DIRECT_GLOBALLY_VALID;
};

}  // namespace il_dataset
