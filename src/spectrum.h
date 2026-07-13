#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "operators.h"

// The k smallest eigenpairs of the generalized problem  C phi = lambda M phi.
//
// Eigenvalues are ascending, lambda_0 ~ 0 on a closed mesh.
// Eigenfunctions are M-orthonormal:  Phi^T M Phi = I.
//
// Requires M to be a lumped (diagonal) mass, stored as a vector of diagonal
// entries. The constructor throws if any entry is non-positive, and if the
// eigensolver fails to converge.
class Spectrum {
public:
    Spectrum(const Operators& ops, int k);

    const Eigen::VectorXd& eigenvalues()    const { return evals_; }  // k
    const Eigen::MatrixXd& eigenfunctions() const { return evecs_; }  // N x k
    const Eigen::VectorXd& mass()           const { return mass_; }   // N (diag of M)
    int size() const { return static_cast<int>(evals_.size()); }

    // Project a scalar field onto the basis:  a = Phi^T M f   (length k).
    Eigen::VectorXd project(const Eigen::VectorXd& f) const;

    // Project a whole field of functions at once:  A = Phi^T M D   (k x E),
    // one column of coefficients per column of D. Used by functional maps.
    Eigen::MatrixXd project(const Eigen::MatrixXd& D) const;

private:
    Eigen::VectorXd evals_;
    Eigen::MatrixXd evecs_;
    Eigen::VectorXd mass_;   // diagonal of M, kept so project() is self-contained
};
