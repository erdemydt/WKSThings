#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "operators.h"

// The k smallest eigenpairs of the generalized problem  C φ = λ M φ.
// Eigenvalues are ascending, λ₀ ≈ 0 on a closed mesh.
// Eigenfunctions are M-orthonormal:  Φᵀ M Φ = I.
class Spectrum {
public:
    Spectrum(const Operators& ops, int k);

    const Eigen::VectorXd& eigenvalues()    const { return evals_; }  // k
    const Eigen::MatrixXd& eigenfunctions() const { return evecs_; }  // N × k
    int size() const { return static_cast<int>(evals_.size()); }

    // Project a scalar field onto the basis:  a = Φᵀ M f   (length k).
    Eigen::VectorXd project(const Eigen::VectorXd& f) const;

private:
    Eigen::VectorXd evals_;
    Eigen::MatrixXd evecs_;
    Eigen::VectorXd mass_;   // kept so project() is self-contained
};