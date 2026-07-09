#pragma once

#include <Eigen/Sparse>
#include "mesh.h"

// The discrete differential operators for the generalized eigenproblem
//   C φ = λ M φ.
struct Operators {
    Eigen::SparseMatrix<double> stiffness;  // C: cotangent, positive semidefinite
    Eigen::SparseMatrix<double> mass;       // M: lumped (barycentric), diagonal
};

Eigen::SparseMatrix<double> build_stiffness(const Mesh& mesh);
Eigen::SparseMatrix<double> build_mass(const Mesh& mesh);

Operators build_operators(const Mesh& mesh);

Operators build_operators_reference(const Mesh& mesh);