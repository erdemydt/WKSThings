#include "mesh.h"
#include "iostream"
#include "string"
#include "operators.h"

int main() {

    try {
        std::string meshName = "meshes/bimba.obj";
        Mesh mesh = Mesh::load(meshName);
        std::cout << "loaded " << meshName << "\n"
            << "  vertices: " << mesh.num_vertices() << "\n"
            << "  faces:    " << mesh.num_faces() << "\n";

        Operators ops1 = build_operators(mesh);
        Operators ops2 = build_operators_reference(mesh);
        Eigen::SparseMatrix<double> dC = ops1.stiffness - ops2.stiffness;
        Eigen::SparseMatrix<double> dM = ops1.mass - ops2.mass;

        auto max_abs = [](const Eigen::SparseMatrix<double>& A) -> double {
            return A.nonZeros() == 0 ? 0.0 : A.coeffs().cwiseAbs().maxCoeff();
            };

        std::cout << "C max diff: " << max_abs(dC)
            << "   (C scale: " << ops2.stiffness.coeffs().cwiseAbs().maxCoeff() << ")\n";
        std::cout << "M max diff: " << max_abs(dM)
            << "   (M scale: " << ops2.mass.coeffs().cwiseAbs().maxCoeff() << ")\n";

        // Independent of libigl: row sums of C must vanish; M must be strictly positive.
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(mesh.num_vertices());
        std::cout << "C row-sum max: " << (ops1.stiffness * ones).cwiseAbs().maxCoeff() << "\n";
        std::cout << "M min diagonal: " << ops1.mass.diagonal().minCoeff() << "\n";
        std::cout << "M total area:   " << ops1.mass.diagonal().sum() << "\n";


    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;

    }
    return 0;
}