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

}  // namespace il_dataset
