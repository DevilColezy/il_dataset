#pragma once

#include <Eigen/Core>
#include <vector>

#include "il_dataset/local_planner/privileged_oracle.hpp"
#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class PrivilegedInterventionOracle;

/// Batch start-goal evaluator used ONLY at scene/task-generation time
/// (never in the 5 Hz / 30 Hz loop).  All geometry queries run on the
/// already-built privileged scene map in C++ (sections XXVIII/XXXIX/LIX);
/// Python only samples candidates and organises the dataset quota.
class TaskGenerationOracle {
public:
    explicit TaskGenerationOracle(const TaskGenerationConfig& config);

    /// Evaluate one candidate on the built scene map.  Does NOT mutate the
    /// oracle's persistent goal-dependent state (cost-to-go / connectivity).
    TaskCandidateResult evaluate(const PrivilegedOracle& oracle,
                                 const Eigen::Vector3d& start,
                                 const Eigen::Vector3d& goal) const;

    /// Evaluate a batch of candidates.
    std::vector<TaskCandidateResult> evaluateCandidates(
        const PrivilegedOracle& oracle,
        const std::vector<Eigen::Vector3d>& starts,
        const std::vector<Eigen::Vector3d>& goals) const;

private:
    TaskGenerationConfig config_;
};

}  // namespace il_dataset
