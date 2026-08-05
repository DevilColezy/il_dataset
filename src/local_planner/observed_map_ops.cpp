#include "il_dataset/local_planner/observed_map_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "il_dataset/local_planner/depth_integrator.hpp"

namespace il_dataset {
namespace {

constexpr double kInf = 1.0e20;

inline std::size_t flat_index(int x, int y, int z, int gy, int gz) {
  return (static_cast<std::size_t>(x) * static_cast<std::size_t>(gy) +
          static_cast<std::size_t>(y)) * static_cast<std::size_t>(gz) +
         static_cast<std::size_t>(z);
}

// Felzenszwalb-Huttenlocher exact squared Euclidean distance transform.
void edt_1d(const std::vector<double>& input,
            std::vector<double>* output,
            std::vector<int>* sites,
            std::vector<double>* boundaries) {
  const int n = static_cast<int>(input.size());
  output->assign(static_cast<std::size_t>(n), kInf);
  if (n == 0) return;

  int first_finite = -1;
  for (int i = 0; i < n; ++i) {
    if (input[static_cast<std::size_t>(i)] < kInf * 0.5) {
      first_finite = i;
      break;
    }
  }
  if (first_finite < 0) return;

  sites->resize(static_cast<std::size_t>(n));
  boundaries->resize(static_cast<std::size_t>(n + 1));
  int k = 0;
  (*sites)[0] = first_finite;
  (*boundaries)[0] = -std::numeric_limits<double>::infinity();
  (*boundaries)[1] = std::numeric_limits<double>::infinity();

  for (int q = first_finite + 1; q < n; ++q) {
    const double fq = input[static_cast<std::size_t>(q)];
    if (fq >= kInf * 0.5) continue;
    double separation = 0.0;
    while (true) {
      const int vk = (*sites)[static_cast<std::size_t>(k)];
      const double fvk = input[static_cast<std::size_t>(vk)];
      separation = ((fq + static_cast<double>(q) * q) -
                    (fvk + static_cast<double>(vk) * vk)) /
                   (2.0 * static_cast<double>(q - vk));
      if (separation > (*boundaries)[static_cast<std::size_t>(k)] || k == 0) {
        break;
      }
      --k;
    }
    ++k;
    (*sites)[static_cast<std::size_t>(k)] = q;
    (*boundaries)[static_cast<std::size_t>(k)] = separation;
    (*boundaries)[static_cast<std::size_t>(k + 1)] =
        std::numeric_limits<double>::infinity();
  }

  k = 0;
  for (int q = 0; q < n; ++q) {
    while ((*boundaries)[static_cast<std::size_t>(k + 1)] < q) ++k;
    const int vk = (*sites)[static_cast<std::size_t>(k)];
    const double delta = static_cast<double>(q - vk);
    (*output)[static_cast<std::size_t>(q)] =
        delta * delta + input[static_cast<std::size_t>(vk)];
  }
}

template <typename IsZero>
std::vector<double> edt_3d(int gx, int gy, int gz, IsZero is_zero) {
  const std::size_t voxel_count = static_cast<std::size_t>(gx) * gy * gz;
  std::vector<double> distance(voxel_count, kInf);
  for (int x = 0; x < gx; ++x) {
    for (int y = 0; y < gy; ++y) {
      for (int z = 0; z < gz; ++z) {
        if (is_zero(x, y, z)) {
          distance[flat_index(x, y, z, gy, gz)] = 0.0;
        }
      }
    }
  }

  std::vector<double> transformed(voxel_count, kInf);
  std::vector<double> line;
  std::vector<double> line_out;
  std::vector<int> sites;
  std::vector<double> boundaries;

  // Z axis (contiguous).
  line.resize(static_cast<std::size_t>(gz));
  for (int x = 0; x < gx; ++x) {
    for (int y = 0; y < gy; ++y) {
      for (int z = 0; z < gz; ++z) {
        line[static_cast<std::size_t>(z)] =
            distance[flat_index(x, y, z, gy, gz)];
      }
      edt_1d(line, &line_out, &sites, &boundaries);
      for (int z = 0; z < gz; ++z) {
        transformed[flat_index(x, y, z, gy, gz)] =
            line_out[static_cast<std::size_t>(z)];
      }
    }
  }
  distance.swap(transformed);

  // Y axis.
  line.resize(static_cast<std::size_t>(gy));
  std::fill(transformed.begin(), transformed.end(), kInf);
  for (int x = 0; x < gx; ++x) {
    for (int z = 0; z < gz; ++z) {
      for (int y = 0; y < gy; ++y) {
        line[static_cast<std::size_t>(y)] =
            distance[flat_index(x, y, z, gy, gz)];
      }
      edt_1d(line, &line_out, &sites, &boundaries);
      for (int y = 0; y < gy; ++y) {
        transformed[flat_index(x, y, z, gy, gz)] =
            line_out[static_cast<std::size_t>(y)];
      }
    }
  }
  distance.swap(transformed);

  // X axis.
  line.resize(static_cast<std::size_t>(gx));
  std::fill(transformed.begin(), transformed.end(), kInf);
  for (int y = 0; y < gy; ++y) {
    for (int z = 0; z < gz; ++z) {
      for (int x = 0; x < gx; ++x) {
        line[static_cast<std::size_t>(x)] =
            distance[flat_index(x, y, z, gy, gz)];
      }
      edt_1d(line, &line_out, &sites, &boundaries);
      for (int x = 0; x < gx; ++x) {
        transformed[flat_index(x, y, z, gy, gz)] =
            line_out[static_cast<std::size_t>(x)];
      }
    }
  }
  return transformed;
}

inline bool in_bounds(int x, int y, int z, int gx, int gy, int gz) {
  return static_cast<unsigned>(x) < static_cast<unsigned>(gx) &&
         static_cast<unsigned>(y) < static_cast<unsigned>(gy) &&
         static_cast<unsigned>(z) < static_cast<unsigned>(gz);
}

}  // namespace

py::tuple build_observed_esdf(py::array_t<std::uint8_t> occupancy,
                              double resolution,
                              double max_distance_m,
                              double vehicle_radius_m) {
  if (resolution <= 0.0 || max_distance_m <= 0.0 || vehicle_radius_m < 0.0) {
    throw std::invalid_argument("invalid observed ESDF metric parameters");
  }
  const py::buffer_info input_info = occupancy.request();
  if (input_info.ndim != 3) {
    throw std::invalid_argument("occupancy must be a 3-D uint8 array");
  }
  const int gx = static_cast<int>(input_info.shape[0]);
  const int gy = static_cast<int>(input_info.shape[1]);
  const int gz = static_cast<int>(input_info.shape[2]);
  if (gx <= 0 || gy <= 0 || gz <= 0) {
    throw std::invalid_argument("occupancy dimensions must be positive");
  }

  py::array_t<float> esdf({gx, gy, gz});
  py::array_t<std::uint8_t> safe_known({gx, gy, gz});
  const auto input = occupancy.unchecked<3>();
  auto esdf_out = esdf.mutable_unchecked<3>();
  auto known_out = safe_known.mutable_unchecked<3>();

  {
    py::gil_scoped_release release;
    const auto known_d2 = edt_3d(gx, gy, gz, [&](int x, int y, int z) {
      return input(x, y, z) == UNKNOWN;
    });
    const auto free_to_occupied_d2 = edt_3d(
        gx, gy, gz, [&](int x, int y, int z) {
          return input(x, y, z) == OCCUPIED;
        });
    const auto occupied_to_free_d2 = edt_3d(
        gx, gy, gz, [&](int x, int y, int z) {
          return input(x, y, z) != OCCUPIED;
        });

    bool has_occupied = false;
    for (int x = 0; x < gx && !has_occupied; ++x) {
      for (int y = 0; y < gy && !has_occupied; ++y) {
        for (int z = 0; z < gz; ++z) {
          if (input(x, y, z) == OCCUPIED) {
            has_occupied = true;
            break;
          }
        }
      }
    }

    for (int x = 0; x < gx; ++x) {
      for (int y = 0; y < gy; ++y) {
        for (int z = 0; z < gz; ++z) {
          const std::size_t index = flat_index(x, y, z, gy, gz);
          // Treat the outside of the rolling grid as unknown as well.  This
          // makes the support erosion conservative on all six map faces.
          const double boundary_voxels = static_cast<double>(std::min(
              {x + 1, gx - x, y + 1, gy - y, z + 1, gz - z}));
          const double unknown_voxels = std::sqrt(known_d2[index]);
          const double known_clearance =
              std::min(boundary_voxels, unknown_voxels) * resolution;
          const bool safe = input(x, y, z) != UNKNOWN &&
                            known_clearance + 1.0e-9 >= vehicle_radius_m;
          known_out(x, y, z) = safe ? 1U : 0U;

          double value = max_distance_m - vehicle_radius_m;
          if (has_occupied) {
            const double outside =
                std::sqrt(free_to_occupied_d2[index]) * resolution;
            const double inside =
                std::sqrt(occupied_to_free_d2[index]) * resolution;
            value = outside - inside - vehicle_radius_m;
            value = std::max(-max_distance_m,
                             std::min(max_distance_m, value));
          }
          esdf_out(x, y, z) = safe ? static_cast<float>(value) : 0.0F;
        }
      }
    }
  }
  return py::make_tuple(std::move(esdf), std::move(safe_known));
}

double sample_known_free_corridor(py::array_t<std::uint8_t> occupancy,
                                  py::array_t<double> origin_world,
                                  double resolution,
                                  py::array_t<double> start_world,
                                  py::array_t<double> end_world,
                                  double radius_m,
                                  double spacing_m,
                                  double min_clearance_m) {
  if (resolution <= 0.0 || spacing_m <= 0.0 || radius_m < 0.0 ||
      min_clearance_m < 0.0) {
    throw std::invalid_argument("invalid corridor metric parameters");
  }
  const py::buffer_info occ_info = occupancy.request();
  const py::buffer_info origin_info = origin_world.request();
  const py::buffer_info start_info = start_world.request();
  const py::buffer_info end_info = end_world.request();
  if (occ_info.ndim != 3 || origin_info.size != 3 || start_info.size != 3 ||
      end_info.size != 3) {
    throw std::invalid_argument("invalid corridor array dimensions");
  }
  const int gx = static_cast<int>(occ_info.shape[0]);
  const int gy = static_cast<int>(occ_info.shape[1]);
  const int gz = static_cast<int>(occ_info.shape[2]);
  const auto occ = occupancy.unchecked<3>();
  const auto origin = origin_world.unchecked<1>();
  const auto start = start_world.unchecked<1>();
  const auto end = end_world.unchecked<1>();

  const double vx = end(0) - start(0);
  const double vy = end(1) - start(1);
  const double vz = end(2) - start(2);
  const double length = std::sqrt(vx * vx + vy * vy + vz * vz);
  if (length < 1.0e-6) return 0.0;

  const double swept_radius = radius_m + min_clearance_m;
  const int radius_voxels =
      std::max(0, static_cast<int>(std::ceil(swept_radius / resolution)));
  std::vector<std::tuple<int, int, int, bool>> offsets;
  std::size_t body_offset_count = 0;
  for (int dx = -radius_voxels; dx <= radius_voxels; ++dx) {
    for (int dy = -radius_voxels; dy <= radius_voxels; ++dy) {
      for (int dz = -radius_voxels; dz <= radius_voxels; ++dz) {
        const double metric = std::sqrt(static_cast<double>(
            dx * dx + dy * dy + dz * dz)) * resolution;
        if (metric <= swept_radius + 1.0e-9) {
          const bool inside_body = metric <= radius_m + 1.0e-9;
          offsets.emplace_back(dx, dy, dz, inside_body);
          if (inside_body) ++body_offset_count;
        }
      }
    }
  }
  if (offsets.empty()) {
    offsets.emplace_back(0, 0, 0, true);
    body_offset_count = 1;
  }

  const int sample_count =
      std::min(200, std::max(2, static_cast<int>(length / spacing_m) + 1));
  std::size_t known_free_count = 0;
  const bool layered_clearance = min_clearance_m > 1.0e-9;
  const std::size_t samples_per_center =
      layered_clearance ? body_offset_count : offsets.size();
  const std::size_t total_count =
      static_cast<std::size_t>(sample_count) * samples_per_center;
  const double inverse_resolution = 1.0 / resolution;

  {
    py::gil_scoped_release release;
    for (int i = 0; i < sample_count; ++i) {
      const double fraction = static_cast<double>(i) /
                              static_cast<double>(sample_count - 1);
      const int cx = static_cast<int>(std::floor(
          (start(0) + fraction * vx - origin(0)) * inverse_resolution));
      const int cy = static_cast<int>(std::floor(
          (start(1) + fraction * vy - origin(1)) * inverse_resolution));
      const int cz = static_cast<int>(std::floor(
          (start(2) + fraction * vz - origin(2)) * inverse_resolution));
      std::size_t body_known_free = 0;
      bool clearance_occupied = false;
      for (const auto& offset : offsets) {
        const int x = cx + std::get<0>(offset);
        const int y = cy + std::get<1>(offset);
        const int z = cz + std::get<2>(offset);
        if (!in_bounds(x, y, z, gx, gy, gz)) continue;
        const std::uint8_t value = occ(x, y, z);
        if (layered_clearance && value == OCCUPIED) {
          clearance_occupied = true;
        }
        if (std::get<3>(offset) && value == FREE) {
          ++body_known_free;
        }
      }
      if (!layered_clearance) {
        known_free_count += body_known_free;
      } else if (!clearance_occupied) {
        known_free_count += body_known_free;
      }
    }
  }
  return total_count == 0
             ? 0.0
             : static_cast<double>(known_free_count) /
                   static_cast<double>(total_count);
}

}  // namespace il_dataset
