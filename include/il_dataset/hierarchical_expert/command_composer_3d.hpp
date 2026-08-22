#pragma once
/// @file   command_composer_3d.hpp
/// @brief  CommandComposer3D — the ONLY place that assembles the final 3D
///         command and the canonical 3D trajectory.
///
/// The planar (2D) planner produces a horizontal PlannerResult; the
/// VerticalController produces the vertical channel.  CommandComposer3D
/// merges them into the single VelocityCommand3D that leaves the expert and
/// becomes the recorded label.  XY and Z are NEVER mixed anywhere else.
///
/// Frame discipline: VelocityCommand3D is BODY/FLU — vx forward, vy left,
/// vz up, yaw_rate CCW-positive — exactly what the Flightmare velocity
/// controller consumes.  No world-frame mixing at this boundary.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"
#include "il_dataset/hierarchical_expert/vertical_controller.hpp"

#include <algorithm>
#include <vector>

namespace il_dataset {
namespace expert {

class CommandComposer3D {
public:
    explicit CommandComposer3D(const Params2D& p) : p_(p) {}

    /// Compose the FINAL 3D BODY/FLU command:
    ///   [planar.vx_body, planar.vy_body, vertical.vz_body, planar.yaw_rate]
    /// All channels are BODY/FLU — no frame mixing.
    VelocityCommand3D compose(const PlannerResult& planar,
                              const VerticalCommand& vertical) const {
        VelocityCommand3D cmd;
        cmd.vx_body = planar.vx_body;
        cmd.vy_body = planar.vy_body;
        cmd.vz_body = vertical.vz_body;
        cmd.yaw_rate = planar.yaw_rate;
        return cmd;
    }

    /// Assemble the canonical 3D trajectory from the planar trajectory and
    /// the vertical prediction (z[i] aligned with points[i]).
    Trajectory3D toTrajectory3D(const PlanarTrajectory& planar,
                                const std::vector<double>& z) const {
        Trajectory3D out;
        if (!planar.valid || z.empty()) return out;
        const size_t n = std::min({planar.points.size(), planar.yaw.size(),
                                   planar.t.size(), z.size()});
        out.points.reserve(n);
        out.yaw.reserve(n);
        out.t.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.points.push_back(Vec3d(planar.points[i].x(),
                                       planar.points[i].y(), z[i]));
            out.yaw.push_back(planar.yaw[i]);
            out.t.push_back(planar.t[i]);
        }
        out.valid = out.points.size() >= 2;
        return out;
    }

private:
    Params2D p_;
};

}  // namespace expert
}  // namespace il_dataset
