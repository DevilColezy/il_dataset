#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <string>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

/// Occupancy grid values (must match the Python/C++ convention).
enum GridValue : std::uint8_t {
    UNKNOWN = 0,
    FREE = 1,
    OCCUPIED = 2,
};

/// Configuration for the causal observed occupancy map.
struct ObservedMapConfig {
    double resolution = 0.10;
    double size_x_m = 12.0;
    double size_y_m = 12.0;
    double size_z_m = 5.0;
    double history_seconds = 8.0;
    double occupied_endpoint_margin_m = 0.05;
    double vehicle_radius_m = 0.30;
    double max_depth_m = 5.0;
    double horizontal_fov_deg = 90.0;
    double esdf_max_distance_m = 5.0;
    double free_space_spacing_m = 0.10;
    int depth_integration_step = 2;
    int rebuild_every_n_frames = 3;
    double recenter_threshold_m = 3.0;
};

/// Causal observed occupancy map with an ESDF and a separate known mask.
///
/// Semantics (section X.1):
///  - Unknown space is NEVER treated as free.  The ESDF field stores the
///    true clearance to the nearest OCCUPIED cell for known-free cells and
///    a non-free value (0.0) for unknown cells; the known mask is stored
///    separately.
///  - Formal checks must combine known == true AND clearance > required.
class ObservedMap {
public:
    explicit ObservedMap(const ObservedMapConfig& config);

    /// Reset the grid centered at `center_world`.  All history is lost.
    void reset(const Eigen::Vector3d& center_world);

    /// Re-center when the drone moved more than `recenter_threshold_m`.
    /// Returns true when a full reset happened.
    bool recenterIfNeeded(const Eigen::Vector3d& center_world);

    /// Integrate a depth image (metres, HxW) into the occupancy grid.
    /// Camera convention: camera frame X=right, Y=down, Z=forward (z is
    /// depth along the optical axis).  `quat_xyzw` is the body->world
    /// rotation quaternion in the standard xyzw layout used by this
    /// package.  All back-projection and ray-casting is done in C++.
    void integrateDepth(const float* depth,
                        int height,
                        int width,
                        const Eigen::Vector3d& cam_pos_world,
                        const Eigen::Vector4d& quat_xyzw,
                        double timestamp_s);

    /// Mark the collision-free vehicle volume around `center_world` as
    /// observed FREE (never downgrades an occupied voxel).
    void markVehicleFreeBubble(const Eigen::Vector3d& center_world,
                               double timestamp_s);

    /// Reset voxels not observed within the history window to UNKNOWN.
    void purgeExpired(double now_s);

    /// Rebuild the ESDF + known mask from the current occupancy grid.
    /// Returns true when a rebuild actually ran (per rebuild cadence).
    bool rebuildEsdf();

    /// Force an ESDF rebuild regardless of cadence (used at reset).
    void forceRebuildEsdf();

    // ── Queries ─────────────────────────────────────────────────────
    bool isKnown(double x, double y, double z) const;
    bool isKnownFree(double x, double y, double z, double min_clearance) const;
    /// Signed clearance at a point (trilinear).  Returns NaN when outside
    /// the grid; unknown cells report 0.0 (never free).
    double esdfValue(double x, double y, double z) const;
    std::uint8_t occupancyAt(double x, double y, double z) const;

    // ── Grid access ────────────────────────────────────────────────
    const std::vector<std::uint8_t>& occupancy() const { return occ_; }
    const std::vector<std::uint8_t>& knownMask() const { return known_mask_; }
    const std::vector<float>& esdf() const { return esdf_; }
    bool esdfBuilt() const { return esdf_built_; }
    int gx() const { return gx_; }
    int gy() const { return gy_; }
    int gz() const { return gz_; }
    const Eigen::Vector3d& origin() const { return origin_world_; }
    double resolution() const { return config_.resolution; }
    int revision() const { return revision_; }
    int esdfRevision() const { return esdf_revision_; }
    const ObservedMapConfig& config() const { return config_; }

    /// World <-> grid conversions.
    Eigen::Vector3d worldToGrid(const Eigen::Vector3d& world) const;
    Eigen::Vector3i worldToGridInt(const Eigen::Vector3d& world) const;
    Eigen::Vector3d gridToWorld(const Eigen::Vector3d& grid) const;

    /// Voxel-count diagnostics (C++ reductions, no Python loops).
    int knownCount() const;
    int occupiedCount() const;
    int freeCount() const;
    int unknownCount() const;

    /// True when the drone position is within the grid bounds.
    bool inBounds(double x, double y, double z) const;

    /// Swept-volume braking-risk check (section XVIII).  Predicts the
    /// braking trajectory (reaction delay at constant velocity, then
    /// deceleration to a stop) and samples it continuously.  Unknown space
    /// counts as unsafe.  `body_radius_m` + `safety_margin_m` define the
    /// swept clearance requirement.
    BrakeRiskResult sweptBrakeRisk(const VehicleState& state,
                                   double reaction_delay_s,
                                   double deceleration_mps2,
                                   double body_radius_m,
                                   double safety_margin_m,
                                   double sample_spacing_m) const;

private:
    std::int64_t indexOf(int ix, int iy, int iz) const;
    void buildEsdfImpl();

    ObservedMapConfig config_;
    std::vector<std::uint8_t> occ_;            // [gx, gy, gz] C-order
    std::vector<double> last_obs_time_;        // [gx, gy, gz]
    std::vector<std::uint8_t> known_mask_;     // [gx, gy, gz] 0=unknown, 1=known
    std::vector<float> esdf_;                  // [gx, gy, gz]
    int gx_ = 0;
    int gy_ = 0;
    int gz_ = 0;
    Eigen::Vector3d origin_world_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d center_world_{Eigen::Vector3d::Zero()};
    int revision_ = 0;
    int esdf_revision_ = -1;
    int frames_since_esdf_ = 0;
    bool esdf_built_ = false;
    bool initialized_ = false;
};

}  // namespace il_dataset
