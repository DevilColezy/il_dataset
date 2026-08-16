#include "il_dataset/hierarchical_expert/scene_profile_generator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

namespace {

/// Band a radius into tiny/small/medium/large using configured thresholds.
inline const char* radiusBand(double r, double tiny_max, double small_max,
                              double medium_max) {
    if (r <= tiny_max) return "tiny";
    if (r <= small_max) return "small";
    if (r <= medium_max) return "medium";
    return "large";
}

/// Legacy density class from obstacle count (manifest compatibility).
inline const char* legacyDensityClass(double count,
                                      const BlueprintGenerationConfig& cfg) {
    if (count <= cfg.density_sparse_max) return "sparse";
    if (count >= cfg.density_dense_min) return "dense";
    return "medium";
}

/// Legacy radius class from max obstacle radius (manifest compatibility).
inline const char* legacyRadiusClass(double max_radius, bool is_empty,
                                     const BlueprintGenerationConfig& cfg) {
    if (is_empty || max_radius <= 0.0) return "none";
    if (max_radius <= cfg.radius_small_max_m) return "small";
    if (max_radius >= cfg.radius_large_min_m) return "large";
    return "medium";
}

/// Fill the legacy BlueprintScene classification fields from the realized
/// obstacle set (manifest / downstream compatibility; the NEW pipeline
/// balances profiles directly, these are report-only).
inline void fillLegacySceneClasses(BlueprintScene& scene,
                                   const BlueprintGenerationConfig& cfg) {
    scene.actual_density_class =
        legacyDensityClass(static_cast<double>(scene.actual_obstacle_count),
                           cfg);
    scene.density_class = scene.actual_density_class;
    double min_r = std::numeric_limits<double>::infinity();
    double max_r = 0.0;
    for (const auto& o : scene.obstacles) {
        min_r = std::min(min_r, o.radius);
        max_r = std::max(max_r, o.radius);
    }
    scene.actual_min_radius_m = scene.obstacles.empty() ? 0.0 : min_r;
    scene.actual_max_radius_m = scene.obstacles.empty() ? 0.0 : max_r;
    scene.actual_radius_class = legacyRadiusClass(scene.actual_max_radius_m,
                                                  scene.is_empty, cfg);
    // With profiles the "planned" class IS the achieved class (report).
    scene.planned_density_class = scene.actual_density_class;
    scene.planned_radius_class = scene.actual_radius_class;
    // Legacy stratum id (3x3): density_idx + 3*radius_idx.
    int density_idx = scene.actual_density_class == "sparse"
                          ? 0
                          : (scene.actual_density_class == "dense" ? 2 : 1);
    int radius_idx = scene.actual_radius_class == "small"
                         ? 0
                         : (scene.actual_radius_class == "large" ? 2 : 1);
    scene.stratum_id = radius_idx * 3 + density_idx;
    scene.count_stratum = density_idx;
    scene.radius_stratum = radius_idx;
}

}  // namespace

SceneProfileGenerator::SceneProfileGenerator(
    const BlueprintGenerationConfig& cfg)
    : cfg_(cfg) {
    buildDefaultCatalog();
    if (!cfg_.profiles.empty()) {
        // User-provided profiles are appended / override by name.
        for (const auto& up : cfg_.profiles) {
            bool replaced = false;
            for (auto& p : profiles_) {
                if (p.name == up.name) {
                    p = up;
                    replaced = true;
                    break;
                }
            }
            if (!replaced) profiles_.push_back(up);
        }
    }
}

void SceneProfileGenerator::buildDefaultCatalog() {
    auto add = [this](const char* name, int cmin, int cmax, double rmin,
                      double rmax, const char* mode, double fixed,
                      SceneStructure st, int clusters, double spread,
                      double passage, double weight,
                      std::vector<std::string> tags) {
        SceneProfile p;
        p.name = name;
        p.count_min = cmin;
        p.count_max = cmax;
        p.radius_min = rmin;
        p.radius_max = rmax;
        p.radius_mode = mode;
        p.fixed_radius = fixed;
        p.structure = st;
        p.cluster_count = clusters;
        p.cluster_spread_m = spread;
        p.passage_width_m = passage;
        p.weight = weight;
        p.tags = std::move(tags);
        profiles_.push_back(p);
    };
    //                    name             cmin cmax  rmin  rmax  mode       fixed  struct             clu spread passage weight tags
    add("empty",             0,   0,  0.10, 1.0, "log_uniform", 0.1, SceneStructure::EMPTY, 0, 0.0, 0.0, 1.0, {"clear"});
    add("sparse_tiny",       1,   4,  0.05, 0.15, "log_uniform", 0.1, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"sparse", "tiny"});
    add("dense_tiny",       20,  30,  0.08, 0.12, "log_uniform", 0.1, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"dense", "tiny"});
    add("sparse_small",      1,   5,  0.15, 0.50, "log_uniform", 0.2, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"sparse", "small"});
    add("dense_small",       8,  16,  0.15, 0.50, "log_uniform", 0.3, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"dense", "small"});
    add("sparse_medium",     1,   4,  0.50, 1.50, "log_uniform", 0.8, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"sparse", "medium"});
    add("dense_medium",      5,  10,  0.50, 1.50, "log_uniform", 0.9, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"dense", "medium"});
    add("large_single",      1,   1,  4.00, 6.00, "log_uniform", 5.0, SceneStructure::CENTRAL_BLOCKER, 0, 0.0, 0.0, 1.0, {"large", "blocker"});
    add("large_sparse",      1,   3,  3.00, 6.00, "log_uniform", 4.0, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"large", "sparse", "blocker"});
    add("mixed_tiny_small", 10,  20,  0.08, 0.50, "log_uniform", 0.3, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"mixed"});
    add("mixed_small_medium", 6, 14,  0.20, 1.50, "log_uniform", 0.7, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"mixed"});
    add("mixed_small_large", 4,  10,  0.30, 5.00, "log_uniform", 2.0, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"mixed", "blocker"});
    add("mixed_all",         8,  18,  0.10, 5.00, "log_uniform", 2.0, SceneStructure::UNIFORM, 0, 0.0, 0.0, 1.0, {"mixed", "blocker"});
    add("clustered",        10,  20,  0.20, 0.80, "log_uniform", 0.4, SceneStructure::CLUSTERED, 3, 4.0, 0.0, 1.0, {"clustered"});
    add("corridor",          6,  12,  0.50, 1.50, "log_uniform", 1.0, SceneStructure::CORRIDOR, 0, 0.0, 2.2, 1.0, {"corridor", "narrow"});
    add("bottleneck",        4,   8,  0.80, 2.50, "log_uniform", 1.6, SceneStructure::BOTTLENECK, 0, 0.0, 1.8, 1.0, {"narrow", "blocker"});
    add("chicane",           6,  12,  0.40, 1.20, "log_uniform", 0.8, SceneStructure::CHICANE, 0, 0.0, 2.4, 1.0, {"narrow", "chicane"});
    add("central_blocker",   1,   4,  2.50, 6.00, "log_uniform", 4.0, SceneStructure::CENTRAL_BLOCKER, 0, 0.0, 0.0, 1.0, {"blocker", "large"});
    add("edge_clutter",      8,  16,  0.30, 1.20, "log_uniform", 0.7, SceneStructure::EDGE_CLUTTER, 0, 0.0, 0.0, 1.0, {"edge"});
}

const SceneProfile* SceneProfileGenerator::findProfile(
    const std::string& name) const {
    for (const auto& p : profiles_) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

const SceneProfile* SceneProfileGenerator::pickProfile(
    std::mt19937_64& rng, const std::map<std::string, double>& deficit_weights) const {
    if (profiles_.empty()) return nullptr;
    // Build effective weights: base profile weight x deficit multiplier.
    std::vector<double> w;
    w.reserve(profiles_.size());
    for (const auto& p : profiles_) {
        double mult = 1.0;
        for (const auto& tag : p.tags) {
            const auto it = deficit_weights.find(tag);
            if (it != deficit_weights.end()) mult *= it->second;
        }
        const auto it = deficit_weights.find(p.name);
        if (it != deficit_weights.end()) mult *= it->second;
        w.push_back(p.weight * mult);
    }
    Rng r(rng());
    const int idx = r.weightedPick(w);
    return &profiles_[static_cast<size_t>(idx)];
}

bool SceneProfileGenerator::placeOne(const BlueprintScene& scene,
                                     const SceneProfile& profile,
                                     const BlueprintGenerationConfig& cfg,
                                     Rng& rng, BlueprintObstacle& out) const {
    const WarehouseGeometry& wh = cfg.warehouse;
    const double m = cfg.boundary_margin_m + cfg.free_cell_surface_clearance_m;

    double r = 0.1;
    if (profile.radius_mode == "fixed") {
        r = profile.fixed_radius;
    } else if (profile.radius_mode == "uniform") {
        r = rng.uniform(profile.radius_min, profile.radius_max);
    } else {  // log_uniform
        const double lmin = std::log(std::max(1e-3, profile.radius_min));
        const double lmax = std::log(std::max(1e-3, profile.radius_max));
        r = std::exp(lmin + rng.uniform(0.0, 1.0) * (lmax - lmin));
    }
    // Clamp to the region's fitting size (never silently shrink below the
    // profile minimum — the caller retries the whole scene on failure).
    if (r > wh.width() * 0.5 || r > wh.height() * 0.5) return false;

    double x = rng.uniform(wh.free_min_x + r + m, wh.free_max_x - r - m);
    double y = rng.uniform(wh.free_min_y + r + m, wh.free_max_y - r - m);
    if (!std::isfinite(x) || !std::isfinite(y)) return false;

    // Boundary check (explicit).
    if (x - r < wh.free_min_x + m || x + r > wh.free_max_x - m ||
        y - r < wh.free_min_y + m || y + r > wh.free_max_y - m) {
        return false;
    }
    // Pairwise surface gap >= min_surface_gap_m.
    const double gap = cfg.min_surface_gap_m;
    for (const auto& e : scene.obstacles) {
        if (std::hypot(x - e.x, y - e.y) < r + e.radius + gap - 1e-9) {
            return false;
        }
    }
    out.x = x;
    out.y = y;
    out.radius = r;
    out.height_m = cfg.obstacle_height_m;
    out.id = static_cast<int>(scene.obstacles.size());
    return true;
}

namespace {

/// Obstacles placed by structured profiles are also validated against the
/// existing set (gap + boundary).  This mirrors placeOne() but with an
/// EXPLICIT candidate centre (used by corridor / chicane / clusters).
inline bool placementValid(const BlueprintScene& scene, double x, double y,
                           double r, const BlueprintGenerationConfig& cfg) {
    const WarehouseGeometry& wh = cfg.warehouse;
    const double m = cfg.boundary_margin_m + cfg.free_cell_surface_clearance_m;
    if (!(x - r >= wh.free_min_x + m && x + r <= wh.free_max_x - m &&
          y - r >= wh.free_min_y + m && y + r <= wh.free_max_y - m)) {
        return false;
    }
    for (const auto& e : scene.obstacles) {
        if (std::hypot(x - e.x, y - e.y) < r + e.radius + cfg.min_surface_gap_m - 1e-9) {
            return false;
        }
    }
    return true;
}

inline void addObstacle(BlueprintScene& scene, double x, double y, double r,
                        double h) {
    BlueprintObstacle ob;
    ob.x = x;
    ob.y = y;
    ob.radius = r;
    ob.height_m = h;
    ob.id = static_cast<int>(scene.obstacles.size());
    scene.obstacles.push_back(ob);
}

}  // namespace

bool SceneProfileGenerator::realizeStructured(
    const SceneProfile& profile, const BlueprintGenerationConfig& cfg,
    Rng& rng, int desired, std::vector<BlueprintObstacle>& out,
    int& placed) const {
    const WarehouseGeometry& wh = cfg.warehouse;
    const double m = cfg.boundary_margin_m + cfg.free_cell_surface_clearance_m;
    const double gap = cfg.min_surface_gap_m;
    const double passage = std::max(
        profile.passage_width_m, cfg.plannerRequiredPassage());
    BlueprintScene sc;
    sc.obstacles = out;  // seed with what we already have

    int attempts = 0;
    const int budget = std::max(1000, cfg.max_scene_generation_attempts * 40);
    const double cx = (wh.free_min_x + wh.free_max_x) * 0.5;
    const double cy = (wh.free_min_y + wh.free_max_y) * 0.5;

    while (static_cast<int>(sc.obstacles.size()) < desired &&
           attempts < budget) {
        ++attempts;
        const double r = profile.radius_mode == "fixed"
                             ? profile.fixed_radius
                             : rng.uniform(profile.radius_min, profile.radius_max);

        double x = 0.0, y = 0.0;
        bool candidate = false;
        switch (profile.structure) {
            case SceneStructure::CLUSTERED: {
                // Obstacles cluster around a small number of cluster
                // centres (also recorded in the metadata).
                const int nc = std::max(1, profile.cluster_count);
                const int ci = rng.uniformInt(0, nc - 1);
                const double ccx = rng.uniform(wh.free_min_x + 3.0, wh.free_max_x - 3.0);
                const double ccy = rng.uniform(wh.free_min_y + 3.0, wh.free_max_y - 3.0);
                const double spread = std::max(1.0, profile.cluster_spread_m);
                const double a = rng.uniform(0.0, 2.0 * M_PI);
                const double rad = spread * std::sqrt(rng.uniform(0.0, 1.0));
                x = ccx + rad * std::cos(a);
                y = ccy + rad * std::sin(a);
                candidate = true;
                break;
            }
            case SceneStructure::CORRIDOR:
            case SceneStructure::BOTTLENECK: {
                // Vertical or horizontal corridor: obstacles live in the two
                // side bands; the central passage of width `passage` is kept
                // obstacle-free.
                const bool horizontal = (rng() & 1) == 0;
                if (horizontal) {
                    const double x_band = rng.uniform(0.0, 1.0) < 0.5
                                              ? wh.free_min_x + m
                                              : wh.free_max_x - m;
                    x = x_band < cx ? rng.uniform(wh.free_min_x + m + r,
                                                  cx - passage * 0.5 - r)
                                    : rng.uniform(cx + passage * 0.5 + r,
                                                  wh.free_max_x - m - r);
                    y = rng.uniform(wh.free_min_y + m + r, wh.free_max_y - m - r);
                } else {
                    const double y_band = rng.uniform(0.0, 1.0) < 0.5
                                              ? wh.free_min_y + m
                                              : wh.free_max_y - m;
                    y = y_band < cy ? rng.uniform(wh.free_min_y + m + r,
                                                  cy - passage * 0.5 - r)
                                    : rng.uniform(cy + passage * 0.5 + r,
                                                  wh.free_max_y - m - r);
                    x = rng.uniform(wh.free_min_x + m + r, wh.free_max_x - m - r);
                }
                candidate = true;
                break;
            }
            case SceneStructure::CHICANE: {
                // Zig-zag around a centre line: alternate lateral offsets.
                const bool horizontal = (rng() & 1) == 0;
                const double along = rng.uniform(wh.free_min_x + m + 1.0,
                                                 wh.free_max_x - m - 1.0);
                const double off = (static_cast<int>(sc.obstacles.size()) % 2 == 0)
                                       ? -1.0
                                       : 1.0;
                const double lat = off * (passage * 0.5 + r + 0.4);
                if (horizontal) {
                    x = along;
                    y = clamp(cy + lat, wh.free_min_y + m + r,
                              wh.free_max_y - m - r);
                } else {
                    y = along;
                    x = clamp(cx + lat, wh.free_min_x + m + r,
                              wh.free_max_x - m - r);
                }
                candidate = true;
                break;
            }
            case SceneStructure::CENTRAL_BLOCKER: {
                // 1..2 large cylinders near the centre.
                x = rng.uniform(cx - 2.0, cx + 2.0);
                y = rng.uniform(cy - 3.0, cy + 3.0);
                candidate = true;
                break;
            }
            case SceneStructure::EDGE_CLUTTER: {
                // Obstacles in a band along the region edges.
                const int side = rng.uniformInt(0, 3);
                const double band = 4.0;
                if (side == 0) {  // left
                    x = rng.uniform(wh.free_min_x + m + r,
                                    wh.free_min_x + m + band);
                    y = rng.uniform(wh.free_min_y + m + r, wh.free_max_y - m - r);
                } else if (side == 1) {  // right
                    x = rng.uniform(wh.free_max_x - m - band,
                                    wh.free_max_x - m - r);
                    y = rng.uniform(wh.free_min_y + m + r, wh.free_max_y - m - r);
                } else if (side == 2) {  // bottom
                    x = rng.uniform(wh.free_min_x + m + r, wh.free_max_x - m - r);
                    y = rng.uniform(wh.free_min_y + m + r,
                                    wh.free_min_y + m + band);
                } else {  // top
                    x = rng.uniform(wh.free_min_x + m + r, wh.free_max_x - m - r);
                    y = rng.uniform(wh.free_max_y - m - band,
                                    wh.free_max_y - m - r);
                }
                candidate = true;
                break;
            }
            default:
                return false;
        }

        if (!candidate || !std::isfinite(x) || !std::isfinite(y)) continue;
        if (!placementValid(sc, x, y, r, cfg)) continue;
        addObstacle(sc, x, y, r, cfg.obstacle_height_m);
    }

    if (static_cast<int>(sc.obstacles.size()) != desired) {
        placed = static_cast<int>(sc.obstacles.size());
        return false;
    }
    out = std::move(sc.obstacles);
    placed = desired;
    return true;
}

SceneMetadata SceneProfileGenerator::computeMetadata(
    const SceneProfile& profile, const BlueprintScene& scene, uint64_t seed,
    int attempts) const {
    SceneMetadata md;
    md.profile = profile.name;
    md.scene_seed = seed;
    md.generation_attempt = attempts;
    md.obstacle_count = static_cast<int>(scene.obstacles.size());
    md.cluster_count = profile.structure == SceneStructure::CLUSTERED
                           ? std::max(0, profile.cluster_count)
                           : 0;
    if (scene.obstacles.empty()) {
        md.radius_min = 0.0;
        md.radius_max = 0.0;
        md.radius_mean = 0.0;
        md.largest_obstacle_radius = 0.0;
        return md;
    }
    double sum = 0.0, minr = std::numeric_limits<double>::infinity(),
           maxr = 0.0;
    for (const auto& o : scene.obstacles) {
        sum += o.radius;
        minr = std::min(minr, o.radius);
        maxr = std::max(maxr, o.radius);
        const char* band = radiusBand(o.radius, 0.15, 0.5, 1.5);
        if (std::string(band) == "tiny") ++md.tiny_count;
        else if (std::string(band) == "small") ++md.small_count;
        else if (std::string(band) == "medium") ++md.medium_count;
        else ++md.large_count;
    }
    md.radius_min = minr;
    md.radius_max = maxr;
    md.radius_mean = sum / static_cast<double>(scene.obstacles.size());
    md.largest_obstacle_radius = maxr;
    const double area = cfg_.warehouse.area();
    md.local_density_proxy =
        area > 1e-9 ? 1000.0 * static_cast<double>(scene.obstacles.size()) / area
                    : 0.0;
    return md;
}

SceneGenerationOutcome SceneProfileGenerator::generate(
    const SceneProfile& profile, uint64_t scene_id, uint64_t seed) const {
    SceneGenerationOutcome out;
    out.scene.scene_id = scene_id;
    out.scene.seed = seed;
    out.scene.profile = profile.name;
    out.metadata = computeMetadata(profile, out.scene, seed, 0);

    // Empty profile: the only legal way to produce a zero-obstacle scene.
    if (profile.count_max == 0) {
        out.scene.is_empty = true;
        out.scene.stratum_id = -1;
        out.scene.count_stratum = -1;
        out.scene.radius_stratum = -1;
        out.scene.planned_density_class = "sparse";
        out.scene.planned_radius_class = "none";
        out.scene.requested_obstacle_count = 0;
        out.scene.actual_obstacle_count = 0;
        out.scene.generation_valid = true;
        out.scene.actual_density_class = "sparse";
        out.scene.actual_radius_class = "none";
        out.success = true;
        out.reason = "ok (empty profile)";
        out.metadata.geometry_valid = true;
        return out;
    }

    const int desired = Rng(seed).uniformInt(
        std::max(0, profile.count_min), std::max(0, profile.count_max));
    out.scene.requested_obstacle_count = desired;

    for (int attempt = 1; attempt <= std::max(1, cfg_.max_scene_generation_attempts);
         ++attempt) {
        // Fresh deterministic attempt seed for placement ONLY.
        const uint64_t attempt_seed =
            seed + static_cast<uint64_t>(attempt) * 0x9E3779B97F4A7C15ull;
        Rng rng(attempt_seed);
        out.scene.obstacles.clear();
        bool ok = true;

        if (profile.structure != SceneStructure::UNIFORM &&
            profile.structure != SceneStructure::EMPTY) {
            int placed = 0;
            ok = realizeStructured(profile, cfg_, rng, desired,
                                   out.scene.obstacles, placed);
        } else {
            const int max_attempts =
                std::max(64, cfg_.max_scene_generation_attempts * 64 *
                                 std::max(1, desired));
            int attempts = 0;
            while (static_cast<int>(out.scene.obstacles.size()) < desired &&
                   attempts < max_attempts) {
                ++attempts;
                BlueprintObstacle ob;
                if (!placeOne(out.scene, profile, cfg_, rng, ob)) continue;
                out.scene.obstacles.push_back(ob);
            }
            ok = (static_cast<int>(out.scene.obstacles.size()) == desired);
        }

        out.scene.actual_obstacle_count =
            static_cast<int>(out.scene.obstacles.size());
        out.attempts = attempt;
        if (ok) {
            // Post-placement boundary + gap re-verification (independent).
            bool valid = true;
            std::string reason;
            const double m = cfg_.boundary_margin_m +
                             cfg_.free_cell_surface_clearance_m;
            for (const auto& a : out.scene.obstacles) {
                if (!cfg_.warehouse.inFree(a.x, a.y, a.radius + m - 1e-9)) {
                    valid = false;
                    reason = "boundary_gap_violation";
                    break;
                }
                for (const auto& b : out.scene.obstacles) {
                    if (a.id == b.id) continue;
                    if (std::hypot(a.x - b.x, a.y - b.y) <
                        a.radius + b.radius + cfg_.min_surface_gap_m - 1e-6) {
                        valid = false;
                        reason = "surface_gap_violation";
                        break;
                    }
                }
                if (!valid) break;
            }
            if (valid) {
                out.scene.generation_valid = true;
                out.scene.failure_reason = "";
                out.metadata = computeMetadata(profile, out.scene, seed,
                                               attempt);
                out.metadata.geometry_valid = true;
                fillLegacySceneClasses(out.scene, cfg_);
                out.success = true;
                out.reason = "ok";
                return out;
            }
            out.scene.generation_valid = false;
            out.scene.failure_reason = reason;
        } else {
            out.scene.generation_valid = false;
            out.scene.failure_reason =
                "profile placement failed (desired=" + std::to_string(desired) +
                " actual=" + std::to_string(out.scene.actual_obstacle_count) +
                ")";
        }
    }
    out.metadata.geometry_valid = false;
    out.metadata.geometry_failure_reason = out.scene.failure_reason;
    out.success = false;
    out.reason = out.scene.failure_reason;
    return out;
}

}  // namespace expert
}  // namespace il_dataset
