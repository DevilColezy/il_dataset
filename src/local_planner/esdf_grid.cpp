#include "il_dataset/local_planner/esdf_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace il_dataset {

ESDFGrid::ESDFGrid() = default;

bool ESDFGrid::setData(const float* data,
                       int gx, int gy, int gz,
                       double origin_x, double origin_y, double origin_z,
                       double resolution) {
    if (data == nullptr || gx <= 0 || gy <= 0 || gz <= 0 || resolution <= 0.0) {
        return false;
    }

    gx_ = gx;
    gy_ = gy;
    gz_ = gz;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    origin_z_ = origin_z;
    resolution_ = resolution;
    inv_resolution_ = 1.0 / resolution;

    size_t total = static_cast<size_t>(gx) * gy * gz;
    data_.resize(total);
    std::memcpy(data_.data(), data, total * sizeof(float));

    // Legacy: no known mask → all voxels known
    has_known_mask_ = false;
    unknown_is_free_ = true;  // legacy behaviour
    known_mask_.clear();

    initialized_ = true;
    return true;
}

bool ESDFGrid::setDataWithMask(const float* data,
                                const uint8_t* known_mask,
                                int gx, int gy, int gz,
                                double origin_x, double origin_y, double origin_z,
                                double resolution,
                                bool unknown_is_free) {
    if (data == nullptr || known_mask == nullptr ||
        gx <= 0 || gy <= 0 || gz <= 0 || resolution <= 0.0) {
        return false;
    }

    gx_ = gx;
    gy_ = gy;
    gz_ = gz;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    origin_z_ = origin_z;
    resolution_ = resolution;
    inv_resolution_ = 1.0 / resolution;
    unknown_is_free_ = unknown_is_free;

    size_t total = static_cast<size_t>(gx) * gy * gz;
    data_.resize(total);
    std::memcpy(data_.data(), data, total * sizeof(float));

    known_mask_.resize(total);
    std::memcpy(known_mask_.data(), known_mask, total * sizeof(uint8_t));
    has_known_mask_ = true;

    initialized_ = true;
    return true;
}

double ESDFGrid::getValue(double x, double y, double z) const {
    if (!initialized_) return -1e6;

    double gx_f = worldToGridX(x);
    double gy_f = worldToGridY(y);
    double gz_f = worldToGridZ(z);

    // Clamp to valid continuous range [0, dim-1]
    if (gx_f < -0.5 || gx_f > gx_ - 0.5 ||
        gy_f < -0.5 || gy_f > gy_ - 0.5 ||
        gz_f < -0.5 || gz_f > gz_ - 0.5) {
        return -1e6;  // outside map = collision
    }

    // Trilinear interpolation
    int ix0 = static_cast<int>(std::floor(gx_f));
    int iy0 = static_cast<int>(std::floor(gy_f));
    int iz0 = static_cast<int>(std::floor(gz_f));
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;

    double wx = gx_f - ix0;
    double wy = gy_f - iy0;
    double wz = gz_f - iz0;

    // Clamp indices to valid range for safe lookup
    ix0 = clampIdx(ix0, gx_);
    ix1 = clampIdx(ix1, gx_);
    iy0 = clampIdx(iy0, gy_);
    iy1 = clampIdx(iy1, gy_);
    iz0 = clampIdx(iz0, gz_);
    iz1 = clampIdx(iz1, gz_);

    double c000 = at(ix0, iy0, iz0);
    double c100 = at(ix1, iy0, iz0);
    double c010 = at(ix0, iy1, iz0);
    double c110 = at(ix1, iy1, iz0);
    double c001 = at(ix0, iy0, iz1);
    double c101 = at(ix1, iy0, iz1);
    double c011 = at(ix0, iy1, iz1);
    double c111 = at(ix1, iy1, iz1);

    double c00 = c000 * (1.0 - wx) + c100 * wx;
    double c01 = c001 * (1.0 - wx) + c101 * wx;
    double c10 = c010 * (1.0 - wx) + c110 * wx;
    double c11 = c011 * (1.0 - wx) + c111 * wx;

    double c0 = c00 * (1.0 - wy) + c10 * wy;
    double c1 = c01 * (1.0 - wy) + c11 * wy;

    return c0 * (1.0 - wz) + c1 * wz;
}

Eigen::Vector3d ESDFGrid::getGradient(double x, double y, double z,
                                      double* clearance_out) const {
    if (!initialized_) {
        if (clearance_out) *clearance_out = -1e6;
        return Eigen::Vector3d::Zero();
    }

    double gx_f = worldToGridX(x);
    double gy_f = worldToGridY(y);
    double gz_f = worldToGridZ(z);

    if (gx_f < 0.0 || gx_f > gx_ - 1.0 ||
        gy_f < 0.0 || gy_f > gy_ - 1.0 ||
        gz_f < 0.0 || gz_f > gz_ - 1.0) {
        if (clearance_out) *clearance_out = -1e6;
        return Eigen::Vector3d::Zero();
    }

    int ix0 = static_cast<int>(std::floor(gx_f));
    int iy0 = static_cast<int>(std::floor(gy_f));
    int iz0 = static_cast<int>(std::floor(gz_f));

    // Use central finite difference on trilinear-interpolated values

    double eps = resolution_ * 0.5;
    double vxp = getValue(x + eps, y, z);
    double vxm = getValue(x - eps, y, z);
    double vyp = getValue(x, y + eps, z);
    double vym = getValue(x, y - eps, z);
    double vzp = getValue(x, y, z + eps);
    double vzm = getValue(x, y, z - eps);

    double v0 = getValue(x, y, z);
    if (clearance_out) *clearance_out = v0;

    Eigen::Vector3d grad;
    grad.x() = (vxp - vxm) / (2.0 * eps);
    grad.y() = (vyp - vym) / (2.0 * eps);
    grad.z() = (vzp - vzm) / (2.0 * eps);

    // If any of the offset samples are outside, zero that component
    if (vxp <= -1e5 || vxm <= -1e5) grad.x() = 0.0;
    if (vyp <= -1e5 || vym <= -1e5) grad.y() = 0.0;
    if (vzp <= -1e5 || vzm <= -1e5) grad.z() = 0.0;

    return grad;
}

bool ESDFGrid::isFree(double x, double y, double z, double min_clearance) const {
    return getValue(x, y, z) > min_clearance;
}

bool ESDFGrid::isKnown(double x, double y, double z) const {
    if (!initialized_) return false;
    // Legacy mode: all in-bounds voxels are known
    if (!has_known_mask_ || unknown_is_free_) {
        double gx_f = worldToGridX(x);
        double gy_f = worldToGridY(y);
        double gz_f = worldToGridZ(z);
        return (gx_f >= -0.5 && gx_f <= gx_ - 0.5 &&
                gy_f >= -0.5 && gy_f <= gy_ - 0.5 &&
                gz_f >= -0.5 && gz_f <= gz_ - 0.5);
    }

    // With known mask: check the nearest voxel
    int ix = static_cast<int>(std::floor(worldToGridX(x)));
    int iy = static_cast<int>(std::floor(worldToGridY(y)));
    int iz = static_cast<int>(std::floor(worldToGridZ(z)));

    if (ix < 0 || ix >= gx_ || iy < 0 || iy >= gy_ || iz < 0 || iz >= gz_)
        return false;  // out of bounds = unknown

    size_t idx = static_cast<size_t>(ix) * gy_ * gz_ +
                 static_cast<size_t>(iy) * gz_ +
                 static_cast<size_t>(iz);
    return known_mask_[idx] != 0;
}

bool ESDFGrid::isKnownFree(double x, double y, double z,
                            double min_clearance) const {
    // If unknown is free (legacy), just check clearance
    if (!has_known_mask_ || unknown_is_free_) {
        return isFree(x, y, z, min_clearance);
    }

    // Must be both known AND have sufficient clearance
    if (!isKnown(x, y, z)) return false;
    return getValue(x, y, z) > min_clearance;
}

}  // namespace il_dataset
