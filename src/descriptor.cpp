#include "descriptor.h"

#include <cmath>
#include <stdexcept>

namespace {

    // The strictly-positive part of the spectrum, shared by both descriptors:
    // the constant mode (lambda_0 ~ 0) is dropped, and the squared eigenfunctions
    // are materialized once. A disconnected mesh has several near-zero modes, so
    // skip all below a relative floor rather than just index 0.
    struct PositiveModes {
        Eigen::VectorXd lambda;   // m strictly positive eigenvalues, ascending
        Eigen::MatrixXd Phi2;     // N x m, phi_k(x)^2 for those modes
    };

    PositiveModes positive_modes(const Spectrum& spec) {
        const Eigen::VectorXd& evals = spec.eigenvalues();
        const Eigen::MatrixXd& evecs = spec.eigenfunctions();
        const int k = spec.size();

        const double thresh = 1e-6 * evals.cwiseAbs().maxCoeff();
        int start = 0;
        while (start < k && evals(start) <= thresh) ++start;

        const int m = k - start;
        if (m < 1) {
            throw std::runtime_error("descriptor: no strictly positive eigenvalues.");
        }
        return { evals.tail(m), evecs.rightCols(m).array().square() };
    }

}  // namespace

Eigen::MatrixXd compute_hks(const Spectrum& spec, int num_times) {
    if (num_times < 1) {
        throw std::runtime_error("compute_hks: num_times must be >= 1.");
    }

    const PositiveModes pm = positive_modes(spec);
    const int m = static_cast<int>(pm.lambda.size());

    // Sun et al. time window, log-spaced. lambda ascending => lambda(0) smallest
    // positive (drives t_max), lambda(m-1) largest (drives t_min).
    const double c = 4.0 * std::log(10.0);
    const double t_min = c / pm.lambda(m - 1);
    const double t_max = c / pm.lambda(0);
    const Eigen::VectorXd log_t =
        Eigen::VectorXd::LinSpaced(num_times, std::log(t_min), std::log(t_max));

    // Weight matrix W (m x T): W(k, j) = exp(-lambda_k * t_j). HKS = Phi2 * W.
    Eigen::MatrixXd W(m, num_times);
    for (int j = 0; j < num_times; ++j) {
        const double t = std::exp(log_t(j));
        W.col(j) = (-pm.lambda.array() * t).exp();
    }

    return pm.Phi2 * W;   // N x T
}

Eigen::MatrixXd compute_wks(const Spectrum& spec, int num_energies) {
    if (num_energies < 1) {
        throw std::runtime_error("compute_wks: num_energies must be >= 1.");
    }

    const PositiveModes pm = positive_modes(spec);
    const int m = static_cast<int>(pm.lambda.size());

    const Eigen::VectorXd log_E = pm.lambda.array().log();   // log energies, ascending
    const double e_lo = log_E(0);
    const double e_hi = log_E(m - 1);

    // sigma from the full log range, then pull the endpoints in by 2 sigma so the
    // Gaussians sit inside the sampled spectrum (paper's parameterization).
    const double sigma = 7.0 * (e_hi - e_lo) / num_energies;
    const double e_min = e_lo + 2.0 * sigma;
    const double e_max = e_hi - 2.0 * sigma;
    if (e_max <= e_min) {
        throw std::runtime_error(
            "compute_wks: num_energies too small for the +/- 2 sigma margins "
            "(need a larger num_energies or a wider spectrum).");
    }
    const Eigen::VectorXd es =
        Eigen::VectorXd::LinSpaced(num_energies, e_min, e_max);

    // T (m x M): T(k, j) = exp(-(e_j - log lambda_k)^2 / 2 sigma^2).
    const double inv_two_sig2 = 1.0 / (2.0 * sigma * sigma);
    Eigen::MatrixXd T(m, num_energies);
    for (int j = 0; j < num_energies; ++j) {
        T.col(j) = (-(log_E.array() - es(j)).square() * inv_two_sig2).exp();
    }

    // Unnormalized WKS = Phi2 * T (N x M), then Ce normalization: divide each
    // energy column by the sum of its Gaussian weights (Ce = 1 / column sum).
    Eigen::MatrixXd wks = pm.Phi2 * T;
    const Eigen::VectorXd col_sum = T.colwise().sum().transpose();   // M
    for (int j = 0; j < num_energies; ++j) {
        wks.col(j) /= col_sum(j);
    }

    return wks;   // N x M
}
