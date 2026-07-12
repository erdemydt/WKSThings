#include "visualizer.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <imgui.h>

#include <Eigen/Dense>

#include <algorithm>
#include <stdexcept>

namespace {

struct ViewState {
    Eigen::MatrixXd left_descriptor;
    Eigen::MatrixXd right_descriptor;
    int num_samples = 0;
    int scale = 0;
    bool left_reverses_scale = false;
    bool right_reverses_scale = false;
    std::string left_label;
    std::string right_label;
    polyscope::SurfaceMesh* left_mesh = nullptr;
    polyscope::SurfaceMesh* right_mesh = nullptr;
};

ViewState g_state;

void show_scale(int scale) {
    const int M = g_state.num_samples;
    const int left_col = g_state.left_reverses_scale ? (M - 1 - scale) : scale;
    const int right_col = g_state.right_reverses_scale ? (M - 1 - scale) : scale;

    g_state.left_mesh->addVertexScalarQuantity(g_state.left_label, g_state.left_descriptor.col(left_col).eval())
        ->setColorMap("viridis")
        ->setEnabled(true);
    g_state.right_mesh->addVertexScalarQuantity(g_state.right_label, g_state.right_descriptor.col(right_col).eval())
        ->setColorMap("viridis")
        ->setEnabled(true);
}

void callback() {
    ImGui::TextUnformatted("Scale: 0 = global, M-1 = local");
    ImGui::TextUnformatted("Use the same spatial scale on both sides.");
    if (ImGui::SliderInt("scale", &g_state.scale, 0, g_state.num_samples - 1)) {
        show_scale(g_state.scale);
    }
}

}  // namespace

void show_descriptor_comparison(const Mesh& mesh,
                                const Eigen::MatrixXd& left_descriptor,
                                const Eigen::MatrixXd& right_descriptor,
                                const std::string& left_label,
                                const std::string& right_label,
                                bool left_reverses_scale,
                                bool right_reverses_scale,
                                int initial_scale) {
    if (left_descriptor.rows() != mesh.num_vertices() || right_descriptor.rows() != mesh.num_vertices()) {
        throw std::invalid_argument("Descriptor row count must match the mesh vertex count.");
    }
    if (left_descriptor.cols() != right_descriptor.cols()) {
        throw std::invalid_argument("Descriptor comparisons require the same number of samples on both sides.");
    }

    const int num_samples = static_cast<int>(left_descriptor.cols());
    if (num_samples <= 0) {
        throw std::invalid_argument("Descriptor comparison requires at least one sample column.");
    }

    g_state.left_descriptor = left_descriptor;
    g_state.right_descriptor = right_descriptor;
    g_state.num_samples = num_samples;
    g_state.left_reverses_scale = left_reverses_scale;
    g_state.right_reverses_scale = right_reverses_scale;
    g_state.left_label = left_label;
    g_state.right_label = right_label;
    g_state.scale = (initial_scale < 0) ? (num_samples / 2) : std::clamp(initial_scale, 0, num_samples - 1);

    const Eigen::MatrixXd& V = mesh.vertices();
    const Eigen::MatrixXi& F = mesh.faces();
    const double width = V.col(0).maxCoeff() - V.col(0).minCoeff();
    const double gap = 1.3 * width;

    Eigen::MatrixXd V_right = V;
    V_right.col(0).array() += gap;

    polyscope::init();
    g_state.left_mesh = polyscope::registerSurfaceMesh((left_label + " (left)").c_str(), V, F);
    g_state.right_mesh = polyscope::registerSurfaceMesh((right_label + " (right)").c_str(), V_right, F);

    show_scale(g_state.scale);
    polyscope::state::userCallback = callback;
    polyscope::show();
}