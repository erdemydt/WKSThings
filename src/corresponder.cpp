#include "corresponder.h"

#include <cmath>
#include <stdexcept>

namespace {

// Z-score each column in place: zero mean, unit std across rows. Columns that are
// essentially constant are left at zero (nothing to discriminate).
void zscore_columns(Eigen::MatrixXd& D) {
    for (int j = 0; j < D.cols(); ++j) {
        const double mean = D.col(j).mean();
        const double sd = std::sqrt((D.col(j).array() - mean).square().sum()
                                    / std::max<int>(1, D.rows() - 1));
        if (sd > 1e-12) D.col(j) = (D.col(j).array() - mean) / sd;
        else            D.col(j).setZero();
    }
}

}  // namespace

Eigen::VectorXi nn_match(const Eigen::MatrixXd& descriptors_a,
                         const Eigen::MatrixXd& descriptors_b,
                         bool normalize) {
    if (descriptors_a.cols() != descriptors_b.cols()) {
        throw std::invalid_argument("nn_match: both fields must have the same descriptor length.");
    }

    // Copy so the caller's descriptors are untouched; normalize each field's
    // columns independently (assumes comparable per-column distributions, which
    // holds for a near-isometric pair of the same object).
    Eigen::MatrixXd A = descriptors_a;
    Eigen::MatrixXd B = descriptors_b;
    if (normalize) {
        zscore_columns(A);
        zscore_columns(B);
    }

    const int na = static_cast<int>(A.rows());
    Eigen::VectorXi match(na);

#pragma omp parallel for
    for (int i = 0; i < na; ++i) {
        int idx = 0;
        (B.rowwise() - A.row(i)).rowwise().squaredNorm().minCoeff(&idx);
        match(i) = idx;
    }
    return match;
}

double exact_hit_rate(const Eigen::VectorXi& match, const Eigen::VectorXi& truth) {
    if (match.size() != truth.size()) {
        throw std::invalid_argument("exact_hit_rate: match and truth must be the same length.");
    }
    if (match.size() == 0) return 0.0;
    const long hits = (match.array() == truth.array()).count();
    return static_cast<double>(hits) / static_cast<double>(match.size());
}
