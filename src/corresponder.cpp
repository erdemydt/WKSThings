#include "corresponder.h"
#include "nanoflann.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <cmath>
#include <stdexcept>

namespace {

    class KDTree {
    public:
        explicit KDTree(const Eigen::MatrixXd& data)
            : data_(data),
            index_(static_cast<int>(data.cols()), *this,
                nanoflann::KDTreeSingleIndexAdaptorParams(40 /*leaf size, matches pyFM*/)) {
            index_.buildIndex();
        }

        // Nearest row of the tree data to `query` (length dim). Returns its row index.
        int nearest(const Eigen::VectorXd& query) const {
            size_t idx = 0;
            double dist2 = 0.0;
            nanoflann::KNNResultSet<double> res(1);
            res.init(&idx, &dist2);
            index_.findNeighbors(res, query.data(), nanoflann::SearchParameters());
            return static_cast<int>(idx);
        }

        // --- nanoflann dataset adaptor interface (operates on data_'s rows) ---
        inline size_t kdtree_get_point_count() const { return data_.rows(); }
        inline double kdtree_get_pt(size_t i, size_t d) const { return data_(i, d); }
        template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }

    private:
        using Adaptor = nanoflann::L2_Simple_Adaptor<double, KDTree>;
        using Index = nanoflann::KDTreeSingleIndexAdaptor<Adaptor, KDTree, -1, int>;
        const Eigen::MatrixXd& data_;
        Index index_;
    };




    // Z-score each column in place: zero mean, unit std across rows. Near-constant
    // columns are zeroed (nothing to discriminate).
    void zscore_columns(Eigen::MatrixXd& D) {
        for (int j = 0; j < D.cols(); ++j) {
            const double mean = D.col(j).mean();
            const double sd = std::sqrt((D.col(j).array() - mean).square().sum()
                / std::max<int>(1, D.rows() - 1));
            if (sd > 1e-12) D.col(j) = (D.col(j).array() - mean) / sd;
            else            D.col(j).setZero();
        }
    }

    // Descriptor coefficients in the eigenbasis: A = Phi^T M D  (k x E), columns
    // optionally z-scored first so every descriptor dimension constrains C comparably.
    Eigen::MatrixXd project_descriptors(const Spectrum& spec, const Eigen::MatrixXd& D,
        bool normalize) {
        if (!normalize) return spec.project(D);
        Eigen::MatrixXd Dn = D;
        zscore_columns(Dn);
        return spec.project(Dn);
    }

    // Solve for C (k x k) minimizing  ||C A - B||^2 + alpha * ||C Lambda_A - Lambda_B C||^2.
    // The commutativity term is separable per entry and the data term couples entries
    // only within a row, so C solves one row at a time:
    //   (A A^T + alpha diag((lambda_B(i) - lambda_A(j))^2)) c_i = A b_i.
    Eigen::MatrixXd solve_C(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        const Eigen::VectorXd& lambda_a, const Eigen::VectorXd& lambda_b,
        double alpha) {
        const int k = static_cast<int>(A.rows());
        const Eigen::MatrixXd AAt = A * A.transpose();   // k x k, shared across rows
        Eigen::MatrixXd C(k, k);
        for (int i = 0; i < k; ++i) {
            Eigen::MatrixXd LHS = AAt;
            for (int j = 0; j < k; ++j) {
                const double d = lambda_b(i) - lambda_a(j);
                LHS(j, j) += alpha * d * d;
            }
            const Eigen::VectorXd RHS = A * B.row(i).transpose();   // k
            C.row(i) = LHS.ldlt().solve(RHS).transpose();
        }
        return C;
    }

    // ICP refinement: alternate (1) recover a point map from C, (2) re-solve C as the
    // orthogonal matrix best aligning the matched bases (mass-weighted Procrustes).
    // Orthogonality encodes the near-isometry assumption. Reuses nn_match for step 1.
    Eigen::MatrixXd refine_C(Eigen::MatrixXd C,
        const Eigen::MatrixXd& Phi_a, const Eigen::MatrixXd& Phi_b,
        const Eigen::VectorXd& mass_a, int num_refine) {
        const int na = static_cast<int>(Phi_a.rows());
        const int k = static_cast<int>(Phi_a.cols());
        const KDTree tree_b(Phi_b);

        for (int iter = 0; iter < num_refine; ++iter) {
            const Eigen::MatrixXd mapped = Phi_a * C.transpose();      // N_A x k

            Eigen::VectorXi nn(na);
#pragma omp parallel for
            for (int v = 0; v < na; ++v) nn(v) = tree_b.nearest(mapped.row(v).transpose());

            Eigen::MatrixXd Phi_b_matched(na, k);
            for (int v = 0; v < na; ++v) Phi_b_matched.row(v) = Phi_b.row(nn(v));

            const Eigen::MatrixXd cov =
                Phi_b_matched.transpose() * (mass_a.asDiagonal() * Phi_a);   // k x k
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
            C = svd.matrixU() * svd.matrixV().transpose();
        }
        return C;
    }

    // Thin nanoflann wrapper: builds a KD-tree over the ROWS of a (n x dim) matrix
    // and answers 1-NN queries. Holds a *reference* to the data, so the matrix it
    // was built on must outlive the tree (true in all our uses).



}  // namespace


Eigen::VectorXi nn_match(const Eigen::MatrixXd& descriptors_a,
    const Eigen::MatrixXd& descriptors_b,
    bool normalize) {
    if (descriptors_a.cols() != descriptors_b.cols()) {
        throw std::invalid_argument("nn_match: descriptor lengths differ.");
    }
    Eigen::MatrixXd A = descriptors_a;
    Eigen::MatrixXd B = descriptors_b;
    if (normalize) { zscore_columns(A); zscore_columns(B); }

    const KDTree tree(B);                 // build once over target rows
    const int na = static_cast<int>(A.rows());
    Eigen::VectorXi match(na);

#pragma omp parallel for
    for (int i = 0; i < na; ++i) {
        match(i) = tree.nearest(A.row(i).transpose());
    }
    return match;
}

Eigen::VectorXi nn_match_brute(const Eigen::MatrixXd& descriptors_a,
    const Eigen::MatrixXd& descriptors_b,
    bool normalize) {
    if (descriptors_a.cols() != descriptors_b.cols()) {
        throw std::invalid_argument("nn_match_brute: descriptor lengths differ.");
    }
    Eigen::MatrixXd A = descriptors_a;
    Eigen::MatrixXd B = descriptors_b;
    if (normalize) { zscore_columns(A); zscore_columns(B); }

    const int na = static_cast<int>(A.rows());
    Eigen::VectorXi match(na);
    const Eigen::VectorXd b_sqnorm = B.rowwise().squaredNorm();   // (nb,)
    const Eigen::MatrixXd G = A * B.transpose();                  // (na, nb)

#pragma omp parallel for
    for (int i = 0; i < na; ++i) {
        int idx = 0;
        (b_sqnorm - 2.0 * G.row(i).transpose()).minCoeff(&idx);
        match(i) = idx;
    }
    return match;
}





Eigen::VectorXi fmap_match(const Spectrum& spec_a,
    const Spectrum& spec_b,
    const Eigen::MatrixXd& descriptors_a,
    const Eigen::MatrixXd& descriptors_b,
    double alpha,
    int num_refine,
    bool normalize) {
    if (descriptors_a.cols() != descriptors_b.cols()) {
        throw std::invalid_argument("fmap_match: both descriptor fields must have the same length.");
    }

    // 1. Descriptors as coefficients in each eigenbasis (k x E).
    const Eigen::MatrixXd A = project_descriptors(spec_a, descriptors_a, normalize);
    const Eigen::MatrixXd B = project_descriptors(spec_b, descriptors_b, normalize);

    // 2. Solve the regularized functional map C (k x k).
    Eigen::MatrixXd C = solve_C(A, B, spec_a.eigenvalues(), spec_b.eigenvalues(), alpha);

    // 3. Optional ICP refinement.
    if (num_refine > 0) {
        C = refine_C(C, spec_a.eigenfunctions(), spec_b.eigenfunctions(),
            spec_a.mass(), num_refine);
    }

    // 4. Recover the point map: send A's basis through C, nearest B vertex.
    const Eigen::MatrixXd mapped = spec_a.eigenfunctions() * C.transpose();  // N_A x k
    return nn_match(mapped, spec_b.eigenfunctions(), false);
}

double exact_hit_rate(const Eigen::VectorXi& match, const Eigen::VectorXi& truth) {
    if (match.size() != truth.size()) {
        throw std::invalid_argument("exact_hit_rate: match and truth must be the same length.");
    }
    if (match.size() == 0) return 0.0;
    const long hits = (match.array() == truth.array()).count();
    return static_cast<double>(hits) / static_cast<double>(match.size());
}
