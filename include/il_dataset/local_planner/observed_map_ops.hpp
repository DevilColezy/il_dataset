#pragma once

#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace il_dataset {

/// Build the conservative observed-space ESDF used by the local planner.
///
/// The implementation uses a separable exact squared Euclidean distance
/// transform in C++.  Unknown support is eroded by vehicle_radius and ESDF
/// values have vehicle_radius subtracted exactly once.
pybind11::tuple build_observed_esdf(
    pybind11::array_t<std::uint8_t> occupancy,
    double resolution,
    double max_distance_m,
    double vehicle_radius_m);

/// Return a conservative observed-corridor score.  The body radius must be
/// known FREE.  When min_clearance_m is positive, the additional outer ring
/// must contain no occupied voxel but may remain unknown, matching the local
/// planner's known-mask plus ESDF-clearance semantics.
double sample_known_free_corridor(
    pybind11::array_t<std::uint8_t> occupancy,
    pybind11::array_t<double> origin_world,
    double resolution,
    pybind11::array_t<double> start_world,
    pybind11::array_t<double> end_world,
    double radius_m,
    double spacing_m,
    double min_clearance_m);

/// V15.3: goal-directed depth-first guide-line search in C++.
///
/// Computes the 2-D macro guide line from `start_world` toward
/// `target_world` on a single height slice of the observed occupancy map
/// (UNKNOWN=0, FREE=1, OCCUPIED=2).  Neighbours are expanded depth-first in
/// order of (remaining Chebyshev distance to the terminal, then per-step
/// cost including the obstacle-distance penalty and the lateral temporal
/// consistency penalty), so the line dives toward the terminal while
/// keeping ESDF clearance and never jumping to the opposite side of an
/// obstacle between consecutive 5 Hz updates.
///
/// `prev_line` is the previous tick's guide line (M,2) world xy, used as a
/// lateral reference: each candidate cell's lateral offset from the goal
/// ray is constrained to [ref - lateral_hard_m, ref + lateral_hard_m]
/// (hard block) with a soft band adding `lateral_cost` per metre beyond
/// `lateral_soft_m`.  Pass an empty (0,2) array to disable the constraint.
///
/// Returns an (N,2) array of world xy points (start -> terminal), or a
/// (0,2) empty array when no line exists.
pybind11::array_t<double> compute_guide_line_2d(
    pybind11::array_t<std::uint8_t> occ2d,
    pybind11::array_t<double> origin_xy,
    double resolution,
    pybind11::array_t<double> start_world,
    pybind11::array_t<double> target_world,
    double unknown_cost,
    int penalty_radius_cells,
    double penalty_gain,
    pybind11::array_t<double> prev_line,
    double lateral_soft_m,
    double lateral_hard_m,
    double lateral_cost);

}  // namespace il_dataset
