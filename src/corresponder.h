#pragma once

#include <Eigen/Core>

#include "spectrum.h"

// Correspondence between two shapes, always returned as a length-N_A vector of
// indices into B: match(i) = j means "vertex i on A corresponds to vertex j on B".
// Both matchers below emit this same type, so the scorer and any visualizer
// consume a correspondence without caring how it was produced.

// --- Baseline: nearest neighbour in descriptor space -----------------------
// For each row of D_A, the nearest row of D_B by Euclidean distance. Brute force,
// O(N_A * N_B * E). If `normalize`, z-score each field's columns first.
Eigen::VectorXi nn_match(const Eigen::MatrixXd& descriptors_a,
                         const Eigen::MatrixXd& descriptors_b,
                         bool normalize = true);

// --- Functional maps (Ovsjanikov et al. 2012) ------------------------------
// Represents the map as a small k x k matrix C in the eigenbasis, solved from the
// descriptors as constraints plus a Laplacian-commutativity regularizer (weight
// `alpha`), optionally refined by `num_refine` ICP iterations, then converted to
// a point-to-point map. Needs both spectra (bases, eigenvalues, mass), unlike NN.
//   alpha       regularization strength (larger = smoother, more diagonal C)
//   num_refine  ICP refinement iterations (0 = plain C)
//   normalize   z-score descriptor columns before projecting (as in NN)
Eigen::VectorXi fmap_match(const Spectrum& spec_a,
                           const Spectrum& spec_b,
                           const Eigen::MatrixXd& descriptors_a,
                           const Eigen::MatrixXd& descriptors_b,
                           double alpha = 1e-3,
                           int num_refine = 10,
                           bool normalize = true);

// --- Scoring ---------------------------------------------------------------
// Fraction of vertices whose match equals the ground-truth index. For a Blender-
// posed copy the ground truth is the identity map: truth = [0, 1, ..., N_A-1].
double exact_hit_rate(const Eigen::VectorXi& match, const Eigen::VectorXi& truth);
