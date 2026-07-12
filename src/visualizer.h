#pragma once

#include <Eigen/Core>

#include <string>

#include "mesh.h"

// Opens an interactive side-by-side comparison for two per-vertex descriptors.
// The slider is expressed as localness in [0, M-1], where 0 = global and
// M-1 = local. Each side can optionally reverse that slider mapping to match
// the descriptor's native scale direction.
void show_descriptor_comparison(const Mesh& mesh,
                                const Eigen::MatrixXd& left_descriptor,
                                const Eigen::MatrixXd& right_descriptor,
                                const std::string& left_label,
                                const std::string& right_label,
                                bool left_reverses_scale,
                                bool right_reverses_scale,
                                int initial_scale = -1);