#pragma once

#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace il_dataset {

/// Occupancy grid constants (must match il_observed_map.py).
enum GridValue : std::uint8_t {
    UNKNOWN = 0,
    FREE = 1,
    OCCUPIED = 2,
};

/// Integrate a back-projected depth point-cloud into an occupancy grid.
///
/// Performs both occupied-endpoint marking and free-space ray-casting
/// in C++ to avoid Python loop overhead.  Modifies occ_grid and
/// last_obs_time in-place.
///
/// @param points_world   N×3 float64  world-frame ray endpoints (already
///                       back-projected & transformed by the caller).
/// @param points_cam_z   N   float64  z-coordinate of each point in the
///                       camera frame (depth); used to skip max-range
///                       (no-return) rays for occupancy marking.
/// @param cam_pos        3   float64  camera position in world frame.
/// @param occ_grid       gx×gy×gz uint8  occupancy grid, modified in-place.
/// @param last_obs_time  gx×gy×gz float64  last-observed timestamps,
///                       modified in-place.
/// @param occ_endpoint_margin  metres subtracted from ray length before
///                       marking free space (keeps occupied surface intact).
/// @param free_space_spacing   metres between consecutive free-space
///                       samples along each ray.
/// @param resolution     voxel side length in metres.
/// @param max_depth_m    sensor maximum range; endpoints at this depth
///                       are treated as no-return (free-space only).
/// @param timestamp_s    monotonic timestamp for last_obs_time updates.
/// @param origin         3   float64  world coordinate of voxel (0,0,0).
/// @param grid_dims      3   int32    [gx, gy, gz] grid dimensions.
/// @return               number of voxels that changed state.
int integrate_depth(py::array_t<double> points_world,
                    py::array_t<double> points_cam_z,
                    py::array_t<double> cam_pos,
                    py::array_t<std::uint8_t> occ_grid,
                    py::array_t<double> last_obs_time,
                    double occ_endpoint_margin,
                    double free_space_spacing,
                    double resolution,
                    double max_depth_m,
                    double timestamp_s,
                    py::array_t<double> origin,
                    py::array_t<int> grid_dims);

}  // namespace il_dataset
