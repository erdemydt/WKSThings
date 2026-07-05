#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <igl/readOBJ.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>

// Cotangent of the angle at the apex, given the two edge vectors u, v
// emanating from that apex. cot = cos/sin = (u·v) / |u×v|.
static double cotangent(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    return u.dot(v) / u.cross(v).norm();
}

int main() {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ("meshes/homer.obj", V, F)) {
        std::cerr << "failed to load mesh\n";
        return 1;
    }
    const int n = V.rows();

    // ---- Build stiffness matrix K (positive semidefinite, eigenvalues >= 0) ----
    // Dirichlet energy convention: K_ii = sum of weights, K_ij = -weight.
    std::vector<Eigen::Triplet<double>> Ktrip;
    // ---- Build barycentric lumped mass matrix M (diagonal) ----
    Eigen::VectorXd Mdiag = Eigen::VectorXd::Zero(n);

    for (int f = 0; f < F.rows(); ++f) {
        int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);
        Eigen::Vector3d p0 = V.row(i0), p1 = V.row(i1), p2 = V.row(i2);

        // Triangle area, for the mass matrix. |cross| / 2.
        double area = 0.5 * (p1 - p0).cross(p2 - p0).norm();
        Mdiag(i0) += area / 3.0;
        Mdiag(i1) += area / 3.0;
        Mdiag(i2) += area / 3.0;

        // For each corner, the cotangent of its angle weights the OPPOSITE edge.
        // corner i0 -> opposite edge (i1, i2), etc.
        // The two edge vectors at a corner both emanate from that corner.
        double c0 = cotangent(p1 - p0, p2 - p0);  // angle at i0, weights edge (i1,i2)
        double c1 = cotangent(p2 - p1, p0 - p1);  // angle at i1, weights edge (i2,i0)
        double c2 = cotangent(p0 - p2, p1 - p2);  // angle at i2, weights edge (i0,i1)

        // Each triangle contributes half its cotangent to the edge weight.
        auto addEdge = [&](int a, int b, double cot) {
            double w = 0.5 * cot;
            Ktrip.emplace_back(a, a, w);
            Ktrip.emplace_back(b, b, w);
            Ktrip.emplace_back(a, b, -w);
            Ktrip.emplace_back(b, a, -w);
            };
        addEdge(i1, i2, c0);
        addEdge(i2, i0, c1);
        addEdge(i0, i1, c2);
    }

    Eigen::SparseMatrix<double> K(n, n);
    K.setFromTriplets(Ktrip.begin(), Ktrip.end());

    Eigen::SparseMatrix<double> M(n, n);
    {
        std::vector<Eigen::Triplet<double>> Mtrip;
        for (int i = 0; i < n; ++i) Mtrip.emplace_back(i, i, Mdiag(i));
        M.setFromTriplets(Mtrip.begin(), Mtrip.end());
    }

    // ---- Verify against libigl ----
    Eigen::SparseMatrix<double> L_ref, M_ref;
    igl::cotmatrix(V, F, L_ref);   // negative-semidefinite convention: K = -L_ref
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M_ref);

    double kErr = (K + L_ref).norm();          // K should equal -L_ref
    double mErr = (M - M_ref).norm();
    std::cout << "K vs -L_ref  abs err = " << kErr
        << "   rel = " << kErr / L_ref.norm() << "\n";
    std::cout << "M vs M_ref   abs err = " << mErr
        << "   rel = " << mErr / M_ref.norm() << "\n";

    polyscope::init();
    polyscope::registerSurfaceMesh("bunny", V, F);
    // Reference point mesh already registered as "bunny".
    auto* ps = polyscope::getSurfaceMesh("bunny");

    // A noisy per-vertex function.
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
    ps->addVertexScalarQuantity("noisy x0", x0);

    // K applied to noise: roughness detector. Large everywhere for noise.
    Eigen::VectorXd Kx = K * x0;
    ps->addVertexScalarQuantity("K * x0 (roughness)", Kx);

    // One implicit smoothing step: solve (M + t K) x = M x0.
    double t = 1;   // smoothing time; larger = smoother
    Eigen::SparseMatrix<double> A = M + t * K;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "factorization failed\n";
        std::cout << "K has NaN: " << (K.coeffs().hasNaN()) << "\n";
        std::cout << "M min diag: " << Mdiag.minCoeff() << "\n";
        std::cout << "zero-mass vertices: " << (Mdiag.array() == 0.0).count() << "\n";
        std::cout << "n = " << n << ", zero-mass = "
            << (Mdiag.array() == 0.0).count() << "\n";
        int firstZero = -1, lastZero = -1, count = 0;
        for (int i = 0; i < n; ++i) {
            if (Mdiag(i) == 0.0) {
                if (firstZero < 0) firstZero = i;
                lastZero = i;
                ++count;
            }
        }
        std::cout << "zeros: " << count << " range [" << firstZero
            << ", " << lastZero << "] of " << n << "\n";
        std::cout << "F max index = " << F.maxCoeff()
          << ", n-1 = " << n - 1 << "\n";
        return 1;
    }
    Eigen::VectorXd xs = solver.solve(M * x0);
    ps->addVertexScalarQuantity("smoothed x", xs);
    polyscope::show();
    return 0;
}