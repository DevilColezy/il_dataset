#pragma once

#include <Eigen/Core>
#include <cstdint>

namespace il_dataset {

class ObservedMap;

/// Strict-semantics ESDF query grid used by the local planner, local path
/// search and validation.
///
/// Semantics (section X.1):
///  - The known mask and the ESDF field are stored separately in the
///    ObservedMap; this facade never conflates them.
///  - Unknown space is never treated as free: `isKnownFree()` requires
///    both the known mask AND a clearance above the threshold.
///  - Out-of-map is always unknown (never free).
///  - `getValue()` returns NaN outside the map / when the ESDF is not
///    built; callers must combine it with `isKnown()`.
class ESDFGrid {
public:
    ESDFGrid() = default;

    /// Point the facade at the observed map.  The map must outlive the grid.
    void setMap(const ObservedMap* map);

    /// Conservative clearance at a world point (trilinear).  NaN when the
    /// map is not built or the point is outside the grid.
    double getValue(double x, double y, double z) const;

    /// ESDF gradient via central differences; `clearance_out` receives the
    /// center value.  Returns zero gradient for NaN centers.
    Eigen::Vector3d getGradient(double x, double y, double z,
                                double* clearance_out = nullptr) const;

    /// Known (observed) space check: all interpolation corners known.
    bool isKnown(double x, double y, double z) const;

    /// Known AND clearance above `min_clearance`.
    bool isKnownFree(double x, double y, double z, double min_clearance) const;

    bool initialized() const { return map_ != nullptr; }
    bool hasKnownMask() const { return map_ != nullptr; }

    double resolution() const;
    double originX() const;
    double originY() const;
    double originZ() const;
    int gx() const;
    int gy() const;
    int gz() const;

private:
    const ObservedMap* map_ = nullptr;
};

}  // namespace il_dataset
