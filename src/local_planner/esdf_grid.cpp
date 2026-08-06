#include "il_dataset/local_planner/esdf_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "il_dataset/local_planner/observed_map.hpp"

namespace il_dataset {

void ESDFGrid::setMap(const ObservedMap* map) { map_ = map; }

double ESDFGrid::resolution() const {
    return map_ != nullptr ? map_->resolution() : 0.0;
}
double ESDFGrid::originX() const { return map_ != nullptr ? map_->origin().x() : 0.0; }
double ESDFGrid::originY() const { return map_ != nullptr ? map_->origin().y() : 0.0; }
double ESDFGrid::originZ() const { return map_ != nullptr ? map_->origin().z() : 0.0; }
int ESDFGrid::gx() const { return map_ != nullptr ? map_->gx() : 0; }
int ESDFGrid::gy() const { return map_ != nullptr ? map_->gy() : 0; }
int ESDFGrid::gz() const { return map_ != nullptr ? map_->gz() : 0; }

double ESDFGrid::getValue(double x, double y, double z) const {
    if (map_ == nullptr) return std::numeric_limits<double>::quiet_NaN();
    return map_->esdfValue(x, y, z);
}

Eigen::Vector3d ESDFGrid::getGradient(double x, double y, double z,
                                      double* clearance_out) const {
    if (clearance_out != nullptr) *clearance_out = getValue(x, y, z);
    if (map_ == nullptr) return Eigen::Vector3d::Zero();
    const double h = std::max(0.02, map_->resolution() * 0.5);
    const double cx = getValue(x, y, z);
    if (!std::isfinite(cx)) return Eigen::Vector3d::Zero();
    const double gx = getValue(x + h, y, z);
    const double gy = getValue(x, y + h, z);
    const double gz = getValue(x, y, z + h);
    Eigen::Vector3d gradient;
    gradient.x() = std::isfinite(gx) ? (gx - cx) / h : 0.0;
    gradient.y() = std::isfinite(gy) ? (gy - cx) / h : 0.0;
    gradient.z() = std::isfinite(gz) ? (gz - cx) / h : 0.0;
    return gradient;
}

bool ESDFGrid::isKnown(double x, double y, double z) const {
    if (map_ == nullptr) return false;
    return map_->isKnown(x, y, z);
}

bool ESDFGrid::isKnownFree(double x, double y, double z,
                           double min_clearance) const {
    if (map_ == nullptr) return false;
    return map_->isKnownFree(x, y, z, min_clearance);
}

}  // namespace il_dataset
