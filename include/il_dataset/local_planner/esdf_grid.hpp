#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace il_dataset {

/// Trilinear ESDF grid with safe out-of-bounds handling.
///
/// IMPORTANT: The ESDF values passed to this grid are assumed to have
/// **already** had the drone_radius subtracted by the Python ESDFBuilder.
/// Therefore clearance <= 0 means the drone body is in collision.
/// The local planner's min_clearance parameter adds a further safety margin
/// on top of this already-inflated ESDF – it is NOT a second drone_radius.
class ESDFGrid {
public:
    ESDFGrid();

    /// Set the ESDF data from a contiguous float32 numpy array.
    /// @param data  pointer to float32 data, shape [gx, gy, gz], row-major (C-order)
    /// @param gx, gy, gz  grid dimensions
    /// @param origin_x, origin_y, origin_z  world coordinate of voxel (0,0,0) corner
    /// @param resolution  voxel size in metres
    /// @return true on success
    bool setData(const float* data,
                 int gx, int gy, int gz,
                 double origin_x, double origin_y, double origin_z,
                 double resolution);

    /// Query ESDF value at world position via trilinear interpolation.
    /// Returns a negative sentinel if the point is outside the map.
    double getValue(double x, double y, double z) const;

    /// Query ESDF gradient at world position via trilinear analytic gradient.
    /// Returns zero vector and negative clearance if outside the map.
    Eigen::Vector3d getGradient(double x, double y, double z, double* clearance_out = nullptr) const;

    /// Check if a point has at least the given clearance.
    bool isFree(double x, double y, double z, double min_clearance = 0.0) const;

    /// Grid dimensions.
    int gx() const { return gx_; }
    int gy() const { return gy_; }
    int gz() const { return gz_; }

    /// World origin (corner of voxel 0,0,0).
    double originX() const { return origin_x_; }
    double originY() const { return origin_y_; }
    double originZ() const { return origin_z_; }

    /// Voxel resolution.
    double resolution() const { return resolution_; }

    /// Whether the ESDF has been initialized with valid data.
    bool initialized() const { return initialized_; }

    /// Memory size in bytes.
    size_t memoryBytes() const { return data_.size() * sizeof(float); }

private:
    /// Clamp integer index to valid range.
    inline int clampIdx(int idx, int max_idx) const {
        return (idx < 0) ? 0 : ((idx >= max_idx) ? max_idx - 1 : idx);
    }

    /// Convert world coordinate to continuous grid index.
    // Grid values are located at voxel centres, while origin_* denotes the
    // corner of voxel (0,0,0).  Subtracting 0.5 maps the first centre to index
    // zero and keeps C++ interpolation consistent with Python A*.
    inline double worldToGridX(double x) const { return (x - origin_x_) * inv_resolution_ - 0.5; }
    inline double worldToGridY(double y) const { return (y - origin_y_) * inv_resolution_ - 0.5; }
    inline double worldToGridZ(double z) const { return (z - origin_z_) * inv_resolution_ - 0.5; }

    /// Safe access to grid value at integer index.
    /// Returns a large negative sentinel if out of bounds.
    inline float at(int ix, int iy, int iz) const {
        if (ix < 0 || ix >= gx_ || iy < 0 || iy >= gy_ || iz < 0 || iz >= gz_)
            return -1e6f;
        return data_[ix * gy_ * gz_ + iy * gz_ + iz];
    }

    std::vector<float> data_;  // [gx, gy, gz] in C-order (x is slowest)
    int gx_ = 0, gy_ = 0, gz_ = 0;
    double origin_x_ = 0.0, origin_y_ = 0.0, origin_z_ = 0.0;
    double resolution_ = 1.0;
    double inv_resolution_ = 1.0;
    bool initialized_ = false;
};

}  // namespace il_dataset
