// Visualize a correspondence by XYZ-color transfer. Choose the matcher on the CLI:
//   ./app meshes/homer.obj meshes/homer_posed.obj nn     (default)
//   ./app meshes/homer.obj meshes/homer_posed.obj fmap
//   ./app meshes/homer.obj meshes/homer_posed.obj gt     (identity ground truth)

#include <Eigen/Core>

#include <iostream>
#include <iomanip>
#include <string>

#include "mesh.h"
#include "operators.h"
#include "spectrum.h"
#include "descriptor.h"
#include "corresponder.h"
#include "visualizer.h"

int main(int argc, char* argv[]) {
    const std::string path_a = (argc >= 2) ? argv[1] : "meshes/homer.obj";
    const std::string path_b = (argc >= 3) ? argv[2] : "meshes/homer_posed.obj";
    const std::string method = (argc >= 4) ? argv[3] : "fmap";
    const int k = 100, M = 100;

    const Mesh a = Mesh::load(path_a);
    const Mesh b = Mesh::load(path_b);

    const int N = a.num_vertices();
    const bool aligned = (b.num_vertices() == N) && (a.faces() == b.faces());

    Eigen::VectorXi match;
    if (method == "gt") {
        if (!aligned) { std::cerr << "gt needs an index-aligned pair.\n"; return 1; }
        match = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
    } else {
        const Spectrum spec_a(build_operators(a), k);
        const Spectrum spec_b(build_operators(b), k);
        const Eigen::MatrixXd wks_a = compute_wks(spec_a, M);
        const Eigen::MatrixXd wks_b = compute_wks(spec_b, M);

        if (method == "fmap") {
            match = fmap_match(spec_a, spec_b, wks_a, wks_b, 1e-3, 10, true);
        } else {
            match = nn_match(wks_a, wks_b, true);
        }
    }

    if (aligned) {
        const Eigen::VectorXi truth = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
        std::cout << std::fixed << std::setprecision(4)
                  << method << " exact-hit rate = " << exact_hit_rate(match, truth) << "\n";
    }

    show_correspondence(a, b, match);
    return 0;
}