#include <Eigen/Sparse>
#include <iostream>
#include <vector>

int main() {
    int n = 4;
    std::vector<Eigen::Triplet<double>> triplets;

    // Build a simple tridiagonal matrix: 2 on diagonal, -1 off-diagonal
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0);
        if (i > 0)     triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
    }
    
    Eigen::SparseMatrix<double> S(n, n);
    S.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "nonzeros = " << S.nonZeros() << "\n";
    std::cout << "dense view:\n" << Eigen::MatrixXd(S) << "\n\n";

    // Sparse solve
    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(S);
    Eigen::VectorXd x = solver.solve(b);

    std::cout << "residual norm = " << (S * x - b).norm() << "\n";
    return 0;
}