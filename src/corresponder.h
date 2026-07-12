#pragma once

#include <Eigen/Core>

// Correspondence between two descriptor fields.
//
// A correspondence is a length-N_A vector of indices into B: match(i) = j means
// "vertex i on mesh A corresponds to vertex j on mesh B". This is the shared
// output type -- a functional-maps matcher would return the same thing, so the
// scorer (and any visualizer) consume matches without caring how they were made.

// Nearest-neighbour match in descriptor space: for each row of D_A, the nearest
// row of D_B by Euclidean distance. Brute force, O(N_A * N_B * E) -- fine for a
// few thousand vertices. If `normalize`, z-score each field's columns first so
// every descriptor sample contributes comparably to the distance (otherwise
// large-magnitude columns dominate; matters a lot for HKS, less for WKS).
Eigen::VectorXi nn_match(const Eigen::MatrixXd& descriptors_a,
                         const Eigen::MatrixXd& descriptors_b,
                         bool normalize = true);

// Fraction of vertices whose match equals the ground-truth index.
// For a Blender-posed copy the ground truth is the identity map, so pass
// truth = [0, 1, ..., N_A-1].
double exact_hit_rate(const Eigen::VectorXi& match, const Eigen::VectorXi& truth);
