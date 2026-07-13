#include "spectrum.h"

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// The lowest-k solution of the *standard* problem Ct psi = lambda psi, ascending.
struct StandardSolution {
    Eigen::VectorXd evals;   // k, ascending
    Eigen::MatrixXd evecs;   // N x k, the psi (not yet mapped back to phi)
};

// Shape and range checks. Throws before any work is done.
void validate(const Operators& ops, int k) {
    const Eigen::Index n = ops.stiffness.rows();
    if (ops.stiffness.cols() != n || ops.mass.size() != n) {
        throw std::runtime_error("Spectrum: C must be square and M must match its size.");
    }
    if (k < 1 || k >= n) {
        throw std::runtime_error("Spectrum: need 1 <= k < N.");
    }
}

// Elementwise M^{-1/2}. With M stored as a diagonal vector this is the whole
// reduction's dependency, so guard positivity here -- a zero or negative mass
// would produce inf/nan silently. (No diagonality or empty-row check needed:
// a vector cannot carry off-diagonal entries, and a zero entry is caught here.)
Eigen::VectorXd inverse_sqrt_mass(const Eigen::VectorXd& mass) {
    if ((mass.array() <= 0.0).any()) {
        throw std::runtime_error("Spectrum: mass matrix has a non-positive entry.");
    }
    return mass.cwiseSqrt().cwiseInverse();
}

// Ct = M^{-1/2} C M^{-1/2}.  Congruent to C under a diagonal scaling, so its
// eigenvalues are exactly the generalized eigenvalues lambda of C phi = lambda M phi.
// Scale in place rather than via diag * sparse * diag: no aliasing question,
// no expression-template temporaries, and the indices never move.
// SparseSymShiftSolve reads only the lower triangle (Uplo = Eigen::Lower),
// so any last-bit asymmetry between Ct(i,j) and Ct(j,i) is irrelevant.
Eigen::SparseMatrix<double> reduce_stiffness(const Eigen::SparseMatrix<double>& C,
                                             const Eigen::VectorXd& msi) {
    Eigen::SparseMatrix<double> Ct = C;   // ColMajor, as Spectra expects
    for (Eigen::Index col = 0; col < Ct.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ct, col); it; ++it) {
            it.valueRef() *= msi(it.row()) * msi(it.col());
        }
    }
    Ct.makeCompressed();
    return Ct;
}

// Lowest-k eigenpairs of Ct via shift-invert, returned ascending.
// Owns the shift sizing, the factorization-retry loop, and the ordering fix --
// everything Spectra-specific lives here so the constructor stays a plain recipe.
StandardSolution solve_lowest(const Eigen::SparseMatrix<double>& Ct, int k) {
    const Eigen::Index n = Ct.rows();

    // Ct's scale is C's scale divided by M's, i.e. ~1e5 larger. A shift must be
    // sized against that, not against 1. Anchor it to the diagonal mean:
    //   |sigma| must beat eps * ||Ct||  (~1e-11 here) for the LU to be stable,
    //   and stay well below lambda_1 (~tens) so 1/(lambda_0 - sigma) still separates.
    // 1e-6 * mean(diag) lands mid-window with several decades of slack on each side.
    double sigma = -1e-6 * Ct.diagonal().mean();

    const int ncv = static_cast<int>(std::min<Eigen::Index>(n, std::max(2 * k + 1, 20)));

    StandardSolution sol;

    // NOTE: SparseSymShiftSolve stores a reference to Ct, and SymEigsShiftSolver
    // stores a reference to the op. Both must outlive the solve. Ct outlives this
    // function (caller owns it); op is rebuilt each attempt and used within it.
    const int max_attempts = 10;
    for (int attempt = 0;; ++attempt) {
        try {
            // The op's set_shift() runs here, inside the solver's constructor:
            // it forms (Ct - sigma I), factors it with Eigen::SparseLU, and throws
            // std::invalid_argument if the factorization fails.
            Spectra::SparseSymShiftSolve<double> op(Ct);
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>>
                eigs(op, k, ncv, sigma);

            eigs.init();
            const Eigen::Index nconv = eigs.compute(Spectra::SortRule::LargestMagn);

            if (eigs.info() != Spectra::CompInfo::Successful || nconv < k) {
                throw std::runtime_error(
                    "Spectrum: eigensolver did not converge; try a larger ncv "
                    "or more iterations.");
            }

            sol.evals = eigs.eigenvalues();
            sol.evecs = eigs.eigenvectors();
            break;
        }
        catch (const std::invalid_argument&) {
            // Factorization failure only. A convergence failure is a runtime_error
            // and propagates out.
            if (attempt + 1 == max_attempts) {
                throw std::runtime_error(
                    "Spectrum: shift-invert factorization failed for every sigma tried; "
                    "C is likely not positive semidefinite (check the sign convention).");
            }
            sigma *= 10.0;
        }
    }

    // Spectra's shift solver back-transforms nu = 1/(lambda - sigma) to lambda and then
    // sorts by LargestAlge, so eigenvalues() arrives descending. Reverse to ascending.
    // Column order is preserved by the later msi scaling, so reversing psi here is
    // equivalent to reversing phi later -- do it once, at the source.
    // Guarded rather than unconditional, in case a future Spectra changes this.
    if (sol.evals.size() > 1 && sol.evals(0) > sol.evals(sol.evals.size() - 1)) {
        sol.evals = sol.evals.reverse().eval();
        sol.evecs = sol.evecs.rowwise().reverse().eval();
    }
    return sol;
}

}  // namespace

Spectrum::Spectrum(const Operators& ops, int k)
    : mass_(ops.mass) {
    validate(ops, k);

    const Eigen::VectorXd msi = inverse_sqrt_mass(ops.mass);
    const Eigen::SparseMatrix<double> Ct = reduce_stiffness(ops.stiffness, msi);
    const StandardSolution sol = solve_lowest(Ct, k);

    evals_ = sol.evals;
    evecs_ = msi.asDiagonal() * sol.evecs;   // phi = M^{-1/2} psi
}

Eigen::VectorXd Spectrum::project(const Eigen::VectorXd& f) const {
    if (f.size() != evecs_.rows()) {
        throw std::runtime_error("Spectrum::project: field length does not match N.");
    }
    // M f is elementwise since M is diagonal:  a = Phi^T (M .* f).
    return evecs_.transpose() * mass_.cwiseProduct(f);
}

Eigen::MatrixXd Spectrum::project(const Eigen::MatrixXd& D) const {
    if (D.rows() != evecs_.rows()) {
        throw std::runtime_error("Spectrum::project: field row count does not match N.");
    }
    // Phi^T M D, with M diagonal: scale each row of D by its mass, then Phi^T.
    return evecs_.transpose() * (mass_.asDiagonal() * D);
}
