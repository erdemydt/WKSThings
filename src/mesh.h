#pragma once

#include <string>
#include <Eigen/Dense>

// Triangle mesh: vertex positions and triangle indices.
// V is N×3 (one vertex per row), F is M×3 (three vertex indices per row).
class Mesh {
public:
    Mesh() = default;
    Mesh(Eigen::MatrixXd V, Eigen::MatrixXi F);

    // Reads a triangle mesh from file (OBJ/OFF/PLY). Throws on failure.
    static Mesh load(const std::string& path);

    const Eigen::MatrixXd& vertices()  const { return V_; }
    const Eigen::MatrixXi& faces()     const { return F_; }

    int num_vertices() const { return static_cast<int>(V_.rows()); }
    int num_faces()    const { return static_cast<int>(F_.rows()); }

private:
    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
};