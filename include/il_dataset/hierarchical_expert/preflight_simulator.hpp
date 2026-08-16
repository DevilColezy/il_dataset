#pragma once
/// @file   preflight_simulator.hpp
/// @brief  Lightweight 2D closed-loop preflight harness that drives the
///         SAME HierarchicalExpert (same params, same target encoding,
///         same FSM) as the real Flightmare collection.
///
/// The only difference is the sensor synthesis stage: the observation
/// patch is produced from 2D ray casting against the (privileged, offline)
/// truth scene — BUT from the SAME camera rig as the runtime path
/// (CameraRig2D: same vehicle-pose + Unity T_BC origin, same camera world
/// orientation INCLUDING R_BC, same FOV / range / ray semantics), so the
/// preflight never fires rays from the drone centre while the runtime
/// fires from the forward-mounted camera.  It is built through
/// Flightmare2DObservation::buildFromRays(), the identical code path the
/// runtime uses after decoding a real depth frame.
///
/// The simulator also owns the truth collision / goal-reached audit:
///   * POINT collision at the current state (drone disk vs cylinder);
///   * CONTINUOUS SWEPT collision of the executed segment prev→new
///     (drone disk swept along the segment);
///   * out-of-bounds of the drone disk w.r.t. the configured region.
/// All three are judge-only (never expert action inputs).  A preflight
/// task is accepted only when it reaches the ORIGINAL goal with no point /
/// swept collision, no out-of-bounds, all 5 Hz update labels valid and
/// within the named qualification tick budget.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"
#include "il_dataset/hierarchical_expert/hierarchical_expert.hpp"
#include "il_dataset/hierarchical_expert/flightmare_2d_observation.hpp"

#include <cstdint>
#include <vector>

namespace il_dataset {
namespace expert {

class PreflightSimulator {
public:
    explicit PreflightSimulator(const Params2D& p)
        : p_(p), expert_(), obs_builder_(p) {}

    /// Configure with the scene (truth) and the global grid anchor.
    void configure(const Scene2D& scene, const Vec2d& min_bounds,
                   const Vec2d& max_bounds);

    /// Reset a preflight episode.  start/goal are world XY, initial_yaw is
    /// the Flightmare yaw.
    void resetTask(const Vec2d& start, const Vec2d& goal,
                   double initial_yaw_fm, uint64_t tick, double flight_z);

    /// One 30 Hz tick.  Synthesizes the observation from the truth scene
    /// with the SAME camera rig, runs the SAME expert, then integrates the
    /// executable command with the shared kinematics and audits the
    /// executed segment continuously (swept collision + boundary).
    struct SimStepResult {
        VehicleState2D state;
        ExpertStepOutput output;
        bool truth_collision = false;
        bool out_of_bounds = false;
        bool goal_reached = false;
    };
    SimStepResult step(uint64_t tick, bool collision_override = false);

    const VehicleState2D& state() const { return state_; }
    const Scene2D& scene() const { return scene_; }
    double flightZ() const { return flight_z_; }

private:
    /// Synthesize the instantaneous FOV patch by ray casting from the
    /// CAMERA (CameraRig2D at the vehicle pose) against the truth scene,
    /// then feed the same ray arrays through Flightmare2DObservation::
    /// buildFromRays (the identical runtime patch code path).
    LocalObservation synthesizePatch(uint64_t tick) const;

    /// True if the drone disk (radius r) swept along segment P0→P1 crosses
    /// the region boundary rectangle.
    bool segmentCrossesBounds(double x0, double y0, double x1, double y1,
                              double r) const;

    Params2D p_;
    Scene2D scene_;
    Vec2d min_bounds_{-20.0, -20.0};
    Vec2d max_bounds_{20.0, 20.0};
    HierarchicalExpert expert_;
    Flightmare2DObservation obs_builder_;
    VehicleState2D state_;
    Task2D task_;
    double flight_z_ = 2.0;
    bool configured_ = false;
};

}  // namespace expert
}  // namespace il_dataset
