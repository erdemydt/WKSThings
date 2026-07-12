// NN correspondence experiment: match a shape to its posed copy on WKS and HKS,
// with and without normalization, and report exact-hit rate against the identity
// ground truth. Answers "does z-scoring help?" empirically.

#include <Eigen/Core>

#include <iostream>
#include <iomanip>
#include <string>

#include "mesh.h"
#include "operators.h"
#include "spectrum.h"
#include "descriptor.h"
#include "corresponder.h"

int main(int argc, char* argv[]) {
    const std::string path_a = (argc >= 2) ? argv[1] : "meshes/homer.obj";
    const std::string path_b = (argc >= 3) ? argv[2] : "meshes/homer_posed.obj";
    const int k = (argc >= 4) ? std::stoi(argv[3]) : 100;
    const int M = 100;

    const Mesh a = Mesh::load(path_a);
    const Mesh b = Mesh::load(path_b);

    if (a.num_vertices() != b.num_vertices() || !(a.faces() == b.faces())) {
        std::cerr << "ERROR: pair is not index-aligned; identity ground truth invalid.\n";
        return 1;
    }
    const int N = a.num_vertices();
    std::cout << "index-aligned pair OK: " << N << " vertices\n\n";

    const Spectrum spec_a(build_operators(a), k);
    const Spectrum spec_b(build_operators(b), k);

    const Eigen::MatrixXd wks_a = compute_wks(spec_a, M);
    const Eigen::MatrixXd wks_b = compute_wks(spec_b, M);
    const Eigen::MatrixXd hks_a = compute_hks(spec_a, M);
    const Eigen::MatrixXd hks_b = compute_hks(spec_b, M);

    // Ground truth: identity map (vertex i <-> i).
    const Eigen::VectorXi truth = Eigen::VectorXi::LinSpaced(N, 0, N - 1);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "descriptor  normalize   exact-hit rate\n";
    std::cout << "---------------------------------------\n";

    auto run = [&](const char* name, const Eigen::MatrixXd& DA,
                   const Eigen::MatrixXd& DB, bool norm) {
        const Eigen::VectorXi match = nn_match(DA, DB, norm);
        const double rate = exact_hit_rate(match, truth);
        std::cout << "  " << std::setw(4) << name << "      "
                  << (norm ? "yes " : "no  ") << "       "
                  << rate << "   (" << static_cast<long>(rate * N) << " / " << N << ")\n";
    };

    run("WKS", wks_a, wks_b, false);
    run("WKS", wks_a, wks_b, true);
    run("HKS", hks_a, hks_b, false);
    run("HKS", hks_a, hks_b, true);

    return 0;
}