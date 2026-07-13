#include "visualizer.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <imgui.h>

#include <Eigen/Dense>

#include <algorithm>
#include <stdexcept>

namespace {

// ---------------------------------------------------------------------------
// show_descriptor_comparison: one mesh, two descriptor fields side by side.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// show_pose_pair: two meshes, HKS + WKS + ground truth on each.
// ---------------------------------------------------------------------------
struct PairState {
    Eigen::MatrixXd hks_a, hks_b, wks_a, wks_b;
    int num_samples = 0;
    int scale = 0;
    polyscope::SurfaceMesh* mesh_a = nullptr;
    polyscope::SurfaceMesh* mesh_b = nullptr;
};

PairState g_pair;

// Re-add both descriptors at the current scale. WKS: col = scale (0 = global).
// HKS: reversed, so the same slider position is the same spatial scale on both.
// setEnabled is applied once (first call) to default to WKS; later re-adds inherit
// the user's toggle choice, so switching to HKS in the panel sticks across slides.
void update_pair(int scale) {
    const int M = g_pair.num_samples;
    const int wks_col = std::clamp(scale, 0, M - 1);
    const int hks_col = (M - 1) - wks_col;

    auto* wa = g_pair.mesh_a->addVertexScalarQuantity("WKS", g_pair.wks_a.col(wks_col).eval())
                   ->setColorMap("viridis");
    auto* wb = g_pair.mesh_b->addVertexScalarQuantity("WKS", g_pair.wks_b.col(wks_col).eval())
                   ->setColorMap("viridis");
    g_pair.mesh_a->addVertexScalarQuantity("HKS", g_pair.hks_a.col(hks_col).eval())->setColorMap("viridis");
    g_pair.mesh_b->addVertexScalarQuantity("HKS", g_pair.hks_b.col(hks_col).eval())->setColorMap("viridis");

    static bool first = true;
    if (first) { wa->setEnabled(true); wb->setEnabled(true); first = false; }
}

void pair_callback() {
    ImGui::TextUnformatted("Left = mesh A,  Right = mesh B (posed).");
    ImGui::TextUnformatted("Scale: 0 = global ... M-1 = local (both same scale).");
    ImGui::TextUnformatted("Toggle HKS / WKS / ground truth in the structure panel.");
    if (ImGui::SliderInt("scale", &g_pair.scale, 0, g_pair.num_samples - 1)) {
        update_pair(g_pair.scale);
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

void show_pose_pair(const Mesh& mesh_a,
                    const Mesh& mesh_b,
                    const Eigen::MatrixXd& hks_a,
                    const Eigen::MatrixXd& hks_b,
                    const Eigen::MatrixXd& wks_a,
                    const Eigen::MatrixXd& wks_b) {
    const int Na = mesh_a.num_vertices();
    const int Nb = mesh_b.num_vertices();
    if (hks_a.rows() != Na || wks_a.rows() != Na || hks_b.rows() != Nb || wks_b.rows() != Nb) {
        throw std::invalid_argument("show_pose_pair: descriptor rows must match each mesh's vertex count.");
    }
    const int M = static_cast<int>(wks_a.cols());
    if (hks_a.cols() != M || wks_b.cols() != M || hks_b.cols() != M || M <= 0) {
        throw std::invalid_argument("show_pose_pair: all descriptors must share the same (positive) sample count.");
    }

    g_pair.hks_a = hks_a;  g_pair.hks_b = hks_b;
    g_pair.wks_a = wks_a;  g_pair.wks_b = wks_b;
    g_pair.num_samples = M;
    g_pair.scale = M / 2;

    // Offset mesh B along x so the two poses sit side by side.
    const Eigen::MatrixXd& Va = mesh_a.vertices();
    const double width = Va.col(0).maxCoeff() - Va.col(0).minCoeff();
    const double gap = 1.3 * width;
    Eigen::MatrixXd Vb = mesh_b.vertices();
    Vb.col(0).array() += gap;

    polyscope::init();
    g_pair.mesh_a = polyscope::registerSurfaceMesh("mesh A", Va, mesh_a.faces());
    g_pair.mesh_b = polyscope::registerSurfaceMesh("mesh B (posed)", Vb, mesh_b.faces());

    // Ground-truth coloring: mesh A's height, mapped by vertex INDEX onto both meshes.
    // Because index i corresponds across the pair, identical color = corresponding
    // point. On B the posed limb carries A's colors, making the map visible directly.
    // Disabled by default (WKS shown first); toggle it on in the panel.
    const Eigen::VectorXd gt = Va.col(1);   // y = height
    g_pair.mesh_a->addVertexScalarQuantity("ground truth (A height)", gt)->setColorMap("turbo");
    g_pair.mesh_b->addVertexScalarQuantity("ground truth (A height)", gt)->setColorMap("turbo");

    update_pair(g_pair.scale);              // creates HKS/WKS, enables WKS
    polyscope::state::userCallback = pair_callback;
    polyscope::show();
}

namespace {

// Per-vertex RGB from position: each axis min..max mapped to 0..1, giving a smooth
// x->R, y->G, z->B field where distinct body parts get distinct colors.
Eigen::MatrixXd position_rgb(const Eigen::MatrixXd& V) {
    Eigen::MatrixXd c(V.rows(), 3);
    for (int d = 0; d < 3; ++d) {
        const double lo = V.col(d).minCoeff();
        const double hi = V.col(d).maxCoeff();
        const double range = (hi > lo) ? (hi - lo) : 1.0;
        c.col(d) = (V.col(d).array() - lo) / range;
    }
    return c;
}

}  // namespace

void show_correspondence(const Mesh& mesh_a,
                         const Mesh& mesh_b,
                         const Eigen::VectorXi& match) {
    const int na = mesh_a.num_vertices();
    const int nb = mesh_b.num_vertices();
    if (static_cast<int>(match.size()) != na) {
        throw std::invalid_argument("show_correspondence: match length must equal mesh A vertex count.");
    }

    const Eigen::MatrixXd& Va = mesh_a.vertices();
    const Eigen::MatrixXd color_a = position_rgb(Va);

    // Push A's colors to B through the map. Gray default reveals B vertices that
    // nothing mapped to; collisions (several i -> same j) let the last one win.
    Eigen::MatrixXd color_b = Eigen::MatrixXd::Constant(nb, 3, 0.6);
    for (int i = 0; i < na; ++i) {
        const int j = match(i);
        if (j >= 0 && j < nb) color_b.row(j) = color_a.row(i);
    }

    const double width = Va.col(0).maxCoeff() - Va.col(0).minCoeff();
    const double gap = 1.3 * width;
    Eigen::MatrixXd Vb = mesh_b.vertices();
    Vb.col(0).array() += gap;

    polyscope::init();
    auto* ma = polyscope::registerSurfaceMesh("A (source colors)", Va, mesh_a.faces());
    auto* mb = polyscope::registerSurfaceMesh("B (transferred)", Vb, mesh_b.faces());
    ma->addVertexColorQuantity("position RGB", color_a)->setEnabled(true);
    mb->addVertexColorQuantity("transferred", color_b)->setEnabled(true);
    polyscope::show();
}