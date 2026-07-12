#include <Eigen/Core>

#include <string>

#include "mesh.h"
#include "operators.h"
#include "spectrum.h"
#include "descriptor.h"
#include "visualizer.h"

int main(int argc, char* argv[]) {
    const std::string path = (argc >= 2) ? argv[1] : "meshes/homer.obj";
    const int k = (argc >= 3) ? std::stoi(argv[2]) : 100;
    const int M = 100;   // descriptor samples

    const Mesh mesh = Mesh::load(path);
    const Operators ops = build_operators(mesh);
    const Spectrum spec(ops, k);

    const Eigen::MatrixXd hks = compute_hks(spec, M);
    const Eigen::MatrixXd wks = compute_wks(spec, M);

    show_descriptor_comparison(mesh, hks, wks, "HKS", "WKS", true, false, M / 2);

    return 0;
}