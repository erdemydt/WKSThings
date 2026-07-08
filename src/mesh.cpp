#include "mesh.h"

#include <stdexcept>
#include <igl/read_triangle_mesh.h>

Mesh::Mesh(Eigen::MatrixXd V, Eigen::MatrixXi F)
    : V_(std::move(V)), F_(std::move(F)) {}

Mesh Mesh::load(const std::string& path) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(path, V, F))
        throw std::runtime_error("Mesh::load: failed to read " + path);
    return Mesh(std::move(V), std::move(F));  // move into the by-value ctor
}