#include "il_dataset/local_planner/observed_map_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
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
          esdf_out(x, y, z) = safe
              ? static_cast<float>(value)
              // ── Fast-Planner aligned: unknown = free ─────────
              // Fast-Planner sets distance_buffer_all_ to 10000 for
              // unobserved voxels.  The ESDF gradient field must be
              // smooth everywhere — an artificial 0 at the known/
              // unknown boundary creates a cliff that blocks the
              // B-spline optimizer.  Instead, assign the maximum
              // distance (empty-space value) so the optimizer sees
              // unknown space as traversable.  The receding-horizon
              // planner will react to newly-observed obstacles within
              // 33 ms (30 Hz), which is ~6 cm of travel at 1.8 m/s.
              : static_cast<float>(max_distance_m - vehicle_radius_m);
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

// ── V15.3: goal-directed DFS guide-line search (C++) ─────────────
py::array_t<double> compute_guide_line_2d(
    py::array_t<std::uint8_t> occ2d,
    py::array_t<double> origin_xy,
    double resolution,
    py::array_t<double> start_world,
    py::array_t<double> target_world,
    double unknown_cost,
    int penalty_radius_cells,
    double penalty_gain,
    py::array_t<double> prev_line,
    double lateral_soft_m,
    double lateral_hard_m,
    double lateral_cost) {
  const py::buffer_info occ_info = occ2d.request();
  const py::buffer_info ori_info = origin_xy.request();
  const py::buffer_info st_info = start_world.request();
  const py::buffer_info tg_info = target_world.request();
  if (occ_info.ndim != 2 || ori_info.size != 2 || st_info.size != 2 ||
      tg_info.size != 2 || resolution <= 0.0 || penalty_radius_cells <= 0) {
    throw std::invalid_argument("compute_guide_line_2d: invalid arguments");
  }
  const int rows = static_cast<int>(occ_info.shape[0]);
  const int cols = static_cast<int>(occ_info.shape[1]);
  if (rows < 3 || cols < 3) {
    throw std::invalid_argument("compute_guide_line_2d: grid too small");
  }
  const auto occ = occ2d.unchecked<2>();
  const auto origin = origin_xy.unchecked<1>();
  const double ox = origin(0), oy = origin(1);
  const auto st = start_world.unchecked<1>();
  const auto tg = target_world.unchecked<1>();

  const auto empty_result = []() -> py::array_t<double> {
    return py::array_t<double>(py::array::ShapeContainer({0, 2}));
  };

  auto world_to_grid = [&](double wx, double wy, int* gx, int* gy) {
    *gx = static_cast<int>(std::floor((wx - ox) / resolution));
    *gy = static_cast<int>(std::floor((wy - oy) / resolution));
  };
  int sx, sy, tx, ty;
  world_to_grid(st(0), st(1), &sx, &sy);
  world_to_grid(tg(0), tg(1), &tx, &ty);
  const auto clamp = [](int& v, int hi) { v = std::max(0, std::min(hi, v)); };
  clamp(sx, rows - 1);
  clamp(sy, cols - 1);
  clamp(tx, rows - 1);
  clamp(ty, cols - 1);

  if (occ(sx, sy) == 2) return empty_result();

  // Terminal occupied -> backtrack toward the start for nearest walkable.
  if (occ(tx, ty) == 2) {
    bool found = false;
    const int steps = std::max(std::abs(tx - sx), std::abs(ty - sy));
    for (int s = 1; s <= steps; ++s) {
      const double f = 1.0 - static_cast<double>(s) / (steps + 1);
      int cx = static_cast<int>(std::lround(sx + (tx - sx) * f));
      int cy = static_cast<int>(std::lround(sy + (ty - sy) * f));
      cx = std::max(0, std::min(rows - 1, cx));
      cy = std::max(0, std::min(cols - 1, cy));
      if (occ(cx, cy) != 2) {
        tx = cx;
        ty = cy;
        found = true;
        break;
      }
    }
    if (!found) return empty_result();
  }

  const std::size_t n_cells = static_cast<std::size_t>(rows) * cols;

  // Multi-source BFS: distance (cells) to the nearest OCCUPIED voxel.
  std::vector<int> dist(n_cells, 1000000);
  std::vector<int> q;
  q.reserve(n_cells);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (occ(i, j) == 2) {
        const std::size_t idx = static_cast<std::size_t>(i) * cols + j;
        dist[idx] = 0;
        q.push_back(static_cast<int>(idx));
      }
    }
  }
  std::size_t head = 0;
  while (head < q.size()) {
    const int idx = q[head++];
    const int i = idx / cols;
    const int j = idx % cols;
    const int nd = dist[static_cast<std::size_t>(idx)] + 1;
    for (int di = -1; di <= 1; ++di) {
      for (int dj = -1; dj <= 1; ++dj) {
        const int ni = i + di, nj = j + dj;
        if (ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;
        const std::size_t nidx = static_cast<std::size_t>(ni) * cols + nj;
        if (dist[nidx] > nd) {
          dist[nidx] = nd;
          q.push_back(static_cast<int>(nidx));
        }
      }
    }
  }

  // Goal direction (world xy) and previous-line lateral reference.
  double gx_w = tg(0) - st(0), gy_w = tg(1) - st(1);
  const double glen = std::hypot(gx_w, gy_w);
  if (glen < 1.0e-9) {
    py::array_t<double> out(py::array::ShapeContainer({1, 2}));
    auto mout = out.mutable_unchecked<2>();
    mout(0, 0) = st(0);
    mout(0, 1) = st(1);
    return out;
  }
  gx_w /= glen;
  gy_w /= glen;

  std::vector<double> ref_f, ref_lat;
  const py::buffer_info pl_info = prev_line.request();
  if (pl_info.ndim == 2 && pl_info.shape[1] == 2 &&
      pl_info.shape[0] > 0) {
    const int m = static_cast<int>(pl_info.shape[0]);
    const auto pl = prev_line.unchecked<2>();
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(static_cast<std::size_t>(m));
    for (int k = 0; k < m; ++k) {
      const double fx = pl(k, 0) - st(0), fy = pl(k, 1) - st(1);
      pairs.emplace_back(gx_w * fx + gy_w * fy, gx_w * fy - gy_w * fx);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<double, double>& a,
                 const std::pair<double, double>& b) {
                return a.first < b.first;
              });
    ref_f.reserve(pairs.size());
    ref_lat.reserve(pairs.size());
    for (const auto& p : pairs) {
      ref_f.push_back(p.first);
      ref_lat.push_back(p.second);
    }
  }
  const auto ref_lateral = [&](double f) -> double {
    if (ref_f.empty()) return 0.0;
    if (ref_f.size() == 1) return ref_lat[0];
    if (f <= ref_f.front()) return ref_lat.front();
    if (f >= ref_f.back()) return ref_lat.back();
    std::size_t lo = 0, hi = ref_f.size() - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (ref_f[mid] <= f) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double f0 = ref_f[lo], f1 = ref_f[hi];
    const double t = (f1 > f0) ? (f - f0) / (f1 - f0) : 0.0;
    return ref_lat[lo] + t * (ref_lat[hi] - ref_lat[lo]);
  };

  // ── Goal-directed weighted A* (dives toward the terminal) ──
  // The guide line is the min-cost path from the drone to the terminal
  // under 8-neighbour step cost = distance + obstacle-distance penalty +
  // lateral temporal penalty.  The heuristic is weighted (w > 1) so the
  // search expands depth-first toward the goal (like a DFS dive) while
  // STILL accumulating the obstacle penalty — a raw DFS cannot accumulate
  // penalty along the path and therefore hugs obstacle boundaries, which
  // made the 30 Hz planner reject every plan (UNKNOWN_SPACE → stutter).
  const int dirs[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1},
                          {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  std::vector<int> parent(n_cells, -1);
  const double kInfCost = 1.0e18;
  std::vector<double> g_score(n_cells, kInfCost);
  const std::size_t start_idx = static_cast<std::size_t>(sx) * cols + sy;
  const std::size_t target_idx = static_cast<std::size_t>(tx) * cols + ty;
  g_score[start_idx] = 0.0;
  const double sqrt2 = std::sqrt(2.0);
  const double h_weight = 1.6;  // >1 -> greedy depth-first dive toward goal

  struct OpenNode {
    double f;
    double neg_g;  // -g: on f ties prefer the DEEPER node (DFS-like dive)
    int idx;
    bool operator>(const OpenNode& o) const {
      if (f != o.f) return f > o.f;
      return neg_g > o.neg_g;
    }
  };
  std::priority_queue<OpenNode, std::vector<OpenNode>,
                      std::greater<OpenNode>> open;
  int best_h = std::max(std::abs(sx - tx), std::abs(sy - ty));
  open.push({h_weight * static_cast<double>(best_h), 0.0,
             static_cast<int>(start_idx)});
  const std::size_t max_expansions =
      std::min<std::size_t>(200000, n_cells * 8);
  std::size_t expansions = 0;
  bool found = false;
  int best_cell = static_cast<int>(start_idx);
  int end_cell = static_cast<int>(start_idx);

  while (!open.empty()) {
    const OpenNode top = open.top();
    open.pop();
    const std::size_t cur = static_cast<std::size_t>(top.idx);
    const double top_g = -top.neg_g;  // neg_g stores -g
    if (top_g > g_score[cur] + 1.0e-9) continue;  // stale entry
    if (++expansions >= max_expansions) break;
    const int ci = top.idx / cols, cj = top.idx % cols;
    if (ci == tx && cj == ty) {
      found = true;
      end_cell = top.idx;
      break;
    }
    const int h_cur = std::max(std::abs(ci - tx), std::abs(cj - ty));
    if (h_cur < best_h) {
      best_h = h_cur;
      best_cell = top.idx;
    }

    for (int d = 0; d < 8; ++d) {
      const int ni = ci + dirs[d][0], nj = cj + dirs[d][1];
      if (ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;
      if (occ(ni, nj) == 2) continue;
      const std::size_t nidx = static_cast<std::size_t>(ni) * cols + nj;

      // Lateral temporal consistency relative to the previous guide line.
      const double wx = ox + (ni + 0.5) * resolution;
      const double wy = oy + (nj + 0.5) * resolution;
      const double fx = wx - st(0), fy = wy - st(1);
      const double f = gx_w * fx + gy_w * fy;
      const double lat = gx_w * fy - gy_w * fx;
      const double dev = std::fabs(lat - ref_lateral(f));
      if (dev > lateral_hard_m) continue;  // hard band: block side flips

      const int d_obs = dist[nidx];
      double step_cost = (occ(ni, nj) == 0 ? unknown_cost : 1.0);
      if (dirs[d][0] != 0 && dirs[d][1] != 0) step_cost *= sqrt2;
      if (d_obs < penalty_radius_cells) {
        step_cost +=
            penalty_gain *
            static_cast<double>(penalty_radius_cells - d_obs);
      }
      if (dev > lateral_soft_m) {
        step_cost += lateral_cost * (dev - lateral_soft_m);
      }
      const double ng = top_g + step_cost;
      if (ng >= g_score[nidx] - 1.0e-9) continue;
      g_score[nidx] = ng;
      parent[nidx] = top.idx;
      const double h = static_cast<double>(
          std::max(std::abs(ni - tx), std::abs(nj - ty)));
      open.push({ng + h_weight * h, -ng, static_cast<int>(nidx)});
    }
  }
  if (!found) end_cell = best_cell;

  // Reconstruct: terminal (or best-progress cell) back to the start.
  std::vector<int> path;
  int c = end_cell;
  while (c >= 0 && path.size() <= n_cells) {
    path.push_back(c);
    if (c == static_cast<int>(start_idx)) break;
    c = parent[static_cast<std::size_t>(c)];
  }
  if (path.empty() || path.back() != static_cast<int>(start_idx)) {
    path.clear();
    path.push_back(static_cast<int>(start_idx));
    if (end_cell != static_cast<int>(start_idx)) {
      path.push_back(end_cell);
    }
  }
  std::reverse(path.begin(), path.end());

  py::array_t<double> out(py::array::ShapeContainer(
      {static_cast<py::ssize_t>(path.size()), 2}));
  auto mout = out.mutable_unchecked<2>();
  for (std::size_t k = 0; k < path.size(); ++k) {
    const int idx = path[k];
    const int i = idx / cols, j = idx % cols;
    mout(static_cast<py::ssize_t>(k), 0) = ox + (i + 0.5) * resolution;
    mout(static_cast<py::ssize_t>(k), 1) = oy + (j + 0.5) * resolution;
  }
  return out;
}

}  // namespace il_dataset
