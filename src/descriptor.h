#pragma once

#include <Eigen/Core>

#include "spectrum.h"

// Spectral descriptors: thin formulas over a solved Spectrum.
//
// Both return an N x E matrix -- row i is vertex i's descriptor, so two rows
// (on the same or different meshes) are directly comparable feature vectors.
// Neither weights by M or normalizes across vertices; per-field normalization
// (z-scoring, scale-invariance) is a corresponder-side concern.

// Heat Kernel Signature.
//
//   HKS(x, t) = sum_{k : lambda_k > 0} exp(-lambda_k t) phi_k(x)^2
//
// The constant mode (lambda_0 ~ 0) is dropped: phi_0 is constant over the
// surface, so its term is the same for every vertex and carries no information.
// Time range: Sun et al. (2009) heuristic, log-spaced,
//   t_min = 4 ln10 / lambda_max,   t_max = 4 ln10 / lambda_1.
Eigen::MatrixXd compute_hks(const Spectrum& spec, int num_times = 100);

// Wave Kernel Signature (Aubry, Schlickewei, Cremers 2011).
//
//   WKS(x, e) = Ce * sum_{k : lambda_k > 0} phi_k(x)^2 * exp(-(e - log lambda_k)^2 / 2 sigma^2)
//   Ce = ( sum_k exp(-(e - log lambda_k)^2 / 2 sigma^2) )^{-1}
//
// Everything lives in log-energy e = log(lambda). Parameters follow the paper:
//   sigma  = 7 * (log lambda_max - log lambda_1) / num_energies
//   e_min  = log lambda_1   + 2 sigma
//   e_max  = log lambda_max - 2 sigma      (endpoints pulled in so the Gaussians fit)
//   e      = num_energies points, linspaced in [e_min, e_max]
//
// The paper computes 300 eigenvalues; here it uses however many the Spectrum
// holds (k). More eigenvalues = more high-frequency detail. num_energies must be
// large enough (> ~28) that the +/- 2 sigma margins leave a non-empty range.
Eigen::MatrixXd compute_wks(const Spectrum& spec, int num_energies = 100);
