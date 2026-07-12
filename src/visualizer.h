#pragma once

#include <Eigen/Core>
#include <string>

#include "mesh.h"

// Show one mesh twice, side by side, each colored by a descriptor field, with a
// shared "scale" slider. Set *_reverses_scale to align two descriptors whose scale
// axes run in opposite directions (HKS vs WKS). initial_scale < 0 -> mid-scale.
void show_descriptor_comparison(const Mesh& mesh,
                                const Eigen::MatrixXd& left_descriptor,
                                const Eigen::MatrixXd& right_descriptor,
                                const std::string& left_label,
                                const std::string& right_label,
                                bool left_reverses_scale,
                                bool right_reverses_scale,
                                int initial_scale);

// Show two DIFFERENT meshes (e.g. a shape and a posed copy) side by side. Each
// carries an HKS and a WKS quantity (toggle in the panel) plus a "ground truth"
// coloring (mesh_a's height, mapped by vertex index onto both). With index-aligned
// ground truth, corresponding points share a color -- so matching color patterns
// across the two poses = a pose-invariant descriptor, the precondition for matching.
// The scale slider drives both descriptors (HKS reversed internally so both sides
// show the same spatial scale). Descriptor columns must match across all four.
void show_pose_pair(const Mesh& mesh_a,
                    const Mesh& mesh_b,
                    const Eigen::MatrixXd& hks_a,
                    const Eigen::MatrixXd& hks_b,
                    const Eigen::MatrixXd& wks_a,
                    const Eigen::MatrixXd& wks_b);