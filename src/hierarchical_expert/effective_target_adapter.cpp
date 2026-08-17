#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"

#include <algorithm>
#include <cmath>

namespace il_dataset {
namespace expert {

int EffectiveTargetAdapter::quantizeBearing(double bearing_rad) const {
    const int n = std::max(3, p_.te_direction_bin_count);
    const double half = binHalfSpanRad();
    const double w = binWidthRad();
    if (!(w > 0.0)) {
        return (n + 1) / 2;  // degenerate config → centre ordinary token
    }
    const double b = clamp(wrapAngle(bearing_rad), -half, half);
    int k = static_cast<int>(std::lround((half - b) / w));
    k = clamp(k, 0, n - 1);
    return k + 1;  // ordinary classes start at 1 (0 = TURN_LEFT)
}

double EffectiveTargetAdapter::tokenCenterBearingRad(int token) const {
    const int n = std::max(3, p_.te_direction_bin_count);
    if (token < 1 || token > n) return 0.0;  // not an ordinary bin
    const int k = token - 1;
    return binHalfSpanRad() - static_cast<double>(k) * binWidthRad();
}

Vec2d EffectiveTargetAdapter::decodeDirectionToken(int token) const {
    const int n = std::max(3, p_.te_direction_bin_count);
    if (token == 0) return turnDirectionBody(SideSelection::LEFT);
    if (token == n + 1) return turnDirectionBody(SideSelection::RIGHT);
    const double b = tokenCenterBearingRad(token);
    return Vec2d(std::cos(b), std::sin(b));
}

double EffectiveTargetAdapter::clampNormalizedDistance(double dist) const {
    const double d = clamp(dist, 0.0, normalMaxDistanceM());
    return d / std::max(1e-9, p_.obs_range_m);
}

double EffectiveTargetAdapter::turnBearingRad(SideSelection side) const {
    const double half = deg2rad(p_.obs_fov_deg) / 2.0;
    const double margin = deg2rad(p_.te_turn_ray_margin_deg);
    return side == SideSelection::LEFT ? half + margin : -(half + margin);
}

Vec2d EffectiveTargetAdapter::turnDirectionBody(SideSelection side) const {
    const double b = turnBearingRad(side);
    return Vec2d(std::cos(b), std::sin(b));
}

EncodedTargetInput EffectiveTargetAdapter::encode(
    const VehicleState2D& state, const Vec2d& original_goal,
    const TargetCorrectionDirective& directive, double goal_z) const {
    EncodedTargetInput out;
    out.valid = true;
    out.source_type = directive.type;
    out.z = goal_z;  // PASS / NORMAL carry the mission altitude
    const double yaw = state.yaw;
    const double R = std::max(1e-9, p_.obs_range_m);
    const double maxd = normalMaxDistanceM();
    const double eps = 1e-6;

    switch (directive.type) {
        case TargetCorrectionType::PASS_THROUGH: {
            // A: PASS_THROUGH — the ORIGINAL goal, truncated to
            // R - reserve along the live goal direction.
            const Vec2d delta = original_goal - state.position;
            const double d = delta.norm();
            if (d <= eps) {
                // Goal (essentially) reached: canonical encoding.
                out.direction_body = Vec2d(1.0, 0.0);
                out.normalized_distance = 0.0;
                out.effective_target_world = state.position;
                out.effective_target_world_valid = true;
                break;
            }
            const Vec2d dir_world = delta / d;
            const double d_clip = std::min(d, maxd);
            out.direction_body = rot2(dir_world, -yaw);
            out.normalized_distance = d_clip / R;
            out.effective_target_world = state.position + dir_world * d_clip;
            out.effective_target_world_valid = true;
            break;
        }
        case TargetCorrectionType::NORMAL_CORRECTION: {
            // B: NORMAL_CORRECTION — the locked corrected_target_world is
            // a FIXED world point for the 5 Hz period.  Every 30 Hz tick
            // the direction/distance are re-derived from the LIVE pose.
            if (!directive.corrected_target_world_valid) {
                // Defensive fallback (never expected): pass through the
                // original goal through the same encoder.
                TargetCorrectionDirective pt;
                pt.type = TargetCorrectionType::PASS_THROUGH;
                pt.valid = true;
                pt.update_event = directive.update_event;
                return encode(state, original_goal, pt, goal_z);
            }
            const Vec2d delta =
                directive.corrected_target_world - state.position;
            const double d = delta.norm();
            if (d <= eps) {
                out.direction_body = Vec2d(1.0, 0.0);
                out.normalized_distance = 0.0;
                out.effective_target_world = directive.corrected_target_world;
                out.effective_target_world_valid = true;
                break;
            }
            const Vec2d dir_world = delta / d;
            const double d_clip = std::min(d, maxd);
            out.direction_body = rot2(dir_world, -yaw);
            out.normalized_distance = d_clip / R;
            out.effective_target_world = state.position + dir_world * d_clip;
            out.effective_target_world_valid = true;
            break;
        }
        case TargetCorrectionType::TURN_LEFT:
        case TargetCorrectionType::TURN_RIGHT: {
            // C: TURN_* — a bounded world-latched direction step.  The
            // fixed world direction is re-expressed in the live body frame
            // every 30 Hz tick, so the bearing converges as the vehicle
            // rotates.  Distance remains EXACTLY 1 (pure rotation).
            const SideSelection side =
                directive.type == TargetCorrectionType::TURN_LEFT
                    ? SideSelection::LEFT
                    : SideSelection::RIGHT;
            Vec2d dir_world;
            if (directive.turn_direction_world_valid) {
                dir_world = directive.turn_direction_world;
                if (dir_world.norm() > eps) {
                    dir_world.normalize();
                } else {
                    dir_world = rot2(turnDirectionBody(side), yaw);
                }
            } else {
                dir_world = rot2(turnDirectionBody(side), yaw);
            }
            out.direction_body = rot2(dir_world, -yaw);
            out.normalized_distance = 1.0;
            out.effective_target_world = state.position + dir_world * R;
            out.effective_target_world_valid = true;
            // Pure rotation: the virtual target stays at the CURRENT
            // altitude, so the 3D effective direction is horizontal.
            out.z = state.z;
            break;
        }
    }
    return out;
}

}  // namespace expert
}  // namespace il_dataset
