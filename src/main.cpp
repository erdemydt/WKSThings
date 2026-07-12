// Verification + curve export driver.
//  - spectrum recap and descriptor sanity (shape, finiteness, smoothness)
//  - writes 1D descriptor curves for a few spread-out points to CSV, to plot
//    externally (reproduces the paper's per-point signature figure).

#include <Eigen/Core>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <cmath>

#include "mesh.h"
#include "operators.h"
#include "spectrum.h"
#include "descriptor.h"

namespace {

// Column-normalized copy (each sample -> zero mean, unit std). Smoothness metric
// only, so HKS's scale spread across time doesn't let a few columns dominate.
Eigen::MatrixXd zscore_columns(const Eigen::MatrixXd& D) {
    Eigen::MatrixXd Z = D;
    for (int j = 0; j < Z.cols(); ++j) {
        const double mean = Z.col(j).mean();
        const double sd = std::sqrt((Z.col(j).array() - mean).square().sum()
                                    / std::max<int>(1, Z.rows() - 1));
        if (sd > 1e-12) Z.col(j) = (Z.col(j).array() - mean) / sd;
        else            Z.col(j).setZero();
    }
    return Z;
}

double avg_edge_dist(const Eigen::MatrixXd& Z, const Eigen::MatrixXi& F) {
    double sum = 0.0; long count = 0;
    for (int f = 0; f < F.rows(); ++f) {
        const int a = F(f, 0), b = F(f, 1), c = F(f, 2);
        sum += (Z.row(a) - Z.row(b)).norm();
        sum += (Z.row(b) - Z.row(c)).norm();
        sum += (Z.row(c) - Z.row(a)).norm();
        count += 3;
    }
    return sum / static_cast<double>(count);
}

double avg_random_dist(const Eigen::MatrixXd& Z, int samples) {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(Z.rows()) - 1);
    double sum = 0.0;
    for (int s = 0; s < samples; ++s)
        sum += (Z.row(pick(rng)) - Z.row(pick(rng))).norm();
    return sum / samples;
}

void validate(const std::string& name, const Eigen::MatrixXd& D,
              const Eigen::MatrixXi& F, int expected_rows) {
    std::cout << "--- " << name << " ---\n";
    std::cout << "  shape            = " << D.rows() << " x " << D.cols()
              << "  (expect " << expected_rows << " x E)\n";
    std::cout << "  all finite?      = " << (D.allFinite() ? "yes" : "NO") << "\n";
    std::cout << "  min / max        = " << D.minCoeff() << " / " << D.maxCoeff() << "\n";

    double min_col_std = std::numeric_limits<double>::infinity();
    for (int j = 0; j < D.cols(); ++j) {
        const double mean = D.col(j).mean();
        const double sd = std::sqrt((D.col(j).array() - mean).square().sum()
                                    / std::max<int>(1, D.rows() - 1));
        min_col_std = std::min(min_col_std, sd);
    }
    std::cout << "  min column std   = " << min_col_std
              << "  (>0 means every sample discriminates)\n";

    const Eigen::MatrixXd Z = zscore_columns(D);
    const double edge = avg_edge_dist(Z, F);
    const double rand = avg_random_dist(Z, 20000);
    std::cout << "  edge dist        = " << edge << "\n";
    std::cout << "  random pair dist = " << rand << "\n";
    std::cout << "  smoothness ratio = " << (edge / rand)
              << "  (<< 1 = spatially smooth)\n\n";
}

// Farthest-point sampling: pick `count` maximally-spread vertices (Euclidean),
// seeded from vertex 0. Gives distinct, reproducible points to plot -- roughly
// the extremities, like the paper's "most distinguished" feature points.
std::vector<int> farthest_point_sample(const Eigen::MatrixXd& V, int count) {
    const int N = static_cast<int>(V.rows());
    std::vector<double> mind(N, std::numeric_limits<double>::infinity());
    std::vector<int> chosen;
    int cur = 0;
    for (int c = 0; c < count; ++c) {
        chosen.push_back(cur);
        for (int i = 0; i < N; ++i) {
            const double d = (V.row(i) - V.row(cur)).squaredNorm();
            if (d < mind[i]) mind[i] = d;
        }
        cur = static_cast<int>(std::max_element(mind.begin(), mind.end()) - mind.begin());
    }
    return chosen;
}

// Write one descriptor's selected rows to CSV: column 0 = sample index, then one
// column per chosen vertex. Rows are the raw descriptor (with Ce for WKS) -- do
// NOT z-score here; that would distort each individual signature's shape.
void write_curves(const std::string& path, const Eigen::MatrixXd& D,
                  const std::vector<int>& pts) {
    std::ofstream f(path);
    f << "sample";
    for (int v : pts) f << ",v" << v;
    f << "\n";
    f << std::setprecision(10);
    for (int j = 0; j < D.cols(); ++j) {
        f << j;
        for (int v : pts) f << "," << D(v, j);
        f << "\n";
    }
    std::cout << "  wrote " << path << " (" << D.cols() << " samples, "
              << pts.size() << " points)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string path = (argc >= 2) ? argv[1] : "meshes/homer.obj";
    const int k = (argc >= 3) ? std::stoi(argv[2]) : 100;

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "=== " << path << "  (k = " << k << ") ===\n";

    const Mesh mesh = Mesh::load(path);
    const int N = mesh.num_vertices();
    std::cout << "V = " << N << "   F = " << mesh.faces().rows() << "\n\n";

    const Operators ops = build_operators(mesh);
    const Spectrum spec(ops, k);

    std::cout << "--- spectrum recap ---\n";
    std::cout << "  lambda[0..3]     = " << spec.eigenvalues().head(4).transpose() << "\n";
    std::cout << "  lambda[k-1]      = " << spec.eigenvalues()(k - 1) << "\n\n";

    const Eigen::MatrixXd H = compute_hks(spec, 100);
    const Eigen::MatrixXd W = compute_wks(spec, 100);

    validate("HKS", H, mesh.faces(), N);
    validate("WKS", W, mesh.faces(), N);

    // -- 1D signature curves for a few spread-out points --
    std::cout << "--- curve export ---\n";
    const std::vector<int> pts = farthest_point_sample(mesh.vertices(), 4);
    std::cout << "  points (FPS): ";
    for (int v : pts) std::cout << v << " ";
    std::cout << "\n";
    write_curves("wks_curves.csv", W, pts);
    write_curves("hks_curves.csv", H, pts);

    return 0;
}