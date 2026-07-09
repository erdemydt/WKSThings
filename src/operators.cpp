#include "operators.h"

#include <vector>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

Eigen::SparseMatrix<double> build_stiffness(const Mesh& mesh) {
    const Eigen::MatrixXd& V = mesh.vertices();
    const Eigen::MatrixXi& F = mesh.faces();
    const int n = mesh.num_vertices();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(F.rows() * 12);  // ~4 entries per angle, 3 angles per face

    for (int f = 0; f < F.rows(); ++f) { // Here we loop over the faces
        int a = F(f, 0), b = F(f, 1), c = F(f, 2); // a, b, c represent the indexes of the points for the face f

        // We get the point coordinates here
        const Eigen::Vector3d pa = V.row(a);
        const Eigen::Vector3d pb = V.row(b);
        const Eigen::Vector3d pc = V.row(c);

        // Cotangent calculation
        const double area = (pb - pa).cross(pc - pa).norm(); // ‖u×v‖ 

        const double cot_a = (pb - pa).dot(pc - pa) / area;
        const double cot_b = (pa - pb).dot(pc - pb) / area;
        const double cot_c = (pa - pc).dot(pb - pc) / area;

        auto add_edge = [&triplets](int i, int j, double w) {
            // Here we build a symmetric matrix with matching diagonal contributions
            triplets.emplace_back(i, j, -w);
            triplets.emplace_back(j, i, -w);
            triplets.emplace_back(i, i, w);
            triplets.emplace_back(j, j, w);
            };

        add_edge(b, c, 0.5 * cot_a);
        add_edge(a, c, 0.5 * cot_b);
        add_edge(a, b, 0.5 * cot_c);

    }

    Eigen::SparseMatrix<double> C(n, n);
    C.setFromTriplets(triplets.begin(), triplets.end());  // sums duplicates
    return C;
}

Eigen::SparseMatrix<double> build_mass(const Mesh& mesh) {
    const Eigen::MatrixXd& V = mesh.vertices();
    const Eigen::MatrixXi& F = mesh.faces();
    const int n = mesh.num_vertices();

    Eigen::VectorXd diag = Eigen::VectorXd::Zero(n);

    for (int f = 0; f < F.rows(); ++f) {
        int a = F(f, 0), b = F(f, 1), c = F(f, 2);

        const Eigen::Vector3d pa = V.row(a);
        const Eigen::Vector3d pb = V.row(b);
        const Eigen::Vector3d pc = V.row(c);

        const double area = 0.5 * (pb - pa).cross(pc - pa).norm();
        const double third = area / 3.0;

        diag(a) += third;
        diag(b) += third;
        diag(c) += third;
    }

    Eigen::SparseMatrix<double> M(n, n);
    M.reserve(n);
    for (int i = 0; i < n; ++i)
        M.insert(i, i) = diag(i);
    M.makeCompressed();
    return M;
}

Operators build_operators(const Mesh& mesh) {
    return { build_stiffness(mesh), build_mass(mesh) };
}

Operators build_operators_reference(const Mesh& mesh) {
    Operators ops;
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(mesh.vertices(), mesh.faces(), L);
    ops.stiffness = -L;  // libigl's L is negative-semidefinite; flip it
    igl::massmatrix(mesh.vertices(), mesh.faces(),
        igl::MASSMATRIX_TYPE_BARYCENTRIC, ops.mass);
    return ops;
}