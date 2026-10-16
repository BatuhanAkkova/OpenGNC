#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <functional>

namespace opengnc {
namespace kalman_filters {

enum class UKFError {
    None,
    CholeskyFailed,
    InvalidParameter
};

/**
 * Generalized Unscented Kalman Filter (UKF).
 * NX: Dimension of the full state vector.
 * NZ: Dimension of the measurement vector.
 * NP: Dimension of the covariance/manifold tangent space (usually NX).
 */
template <int NX, int NZ, int NP = NX>
class UKF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using StateVec = Eigen::Matrix<double, NX, 1>;
    using MeasVec = Eigen::Matrix<double, NZ, 1>;
    using ErrorVec = Eigen::Matrix<double, NP, 1>;
    using CovMat = Eigen::Matrix<double, NP, NP>;
    using SigmaPoints = Eigen::Matrix<double, NX, 2 * NP + 1>;

    // Manifold operations
    using SubtractFunc = std::function<ErrorVec(const StateVec&, const StateVec&)>;
    using AddFunc = std::function<StateVec(const StateVec&, const ErrorVec&)>;
    using MeanFunc = std::function<StateVec(const SigmaPoints&, const Eigen::Matrix<double, 2 * NP + 1, 1>&)>;

    // Dynamics and Measurement models
    using FxFunc = std::function<StateVec(const StateVec&, double)>;
    using HxFunc = std::function<MeasVec(const StateVec&)>;

    UKF(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : alpha(alpha), beta(beta), kappa(kappa), last_error(UKFError::None) {
        
        lambda = alpha * alpha * (NP + kappa) - NP;
        gamma = std::sqrt(NP + lambda);
        
        const int num_sigmas = 2 * NP + 1;
        Wm.setZero();
        Wc.setZero();
        
        Wm(0) = lambda / (NP + lambda);
        Wc(0) = Wm(0) + (1.0 - alpha * alpha + beta);
        
        double w = 1.0 / (2.0 * (NP + lambda));
        for (int i = 1; i < num_sigmas; ++i) {
            Wm(i) = w;
            Wc(i) = w;
        }

        // Default vector space operations
        subtract_x = [](const StateVec& a, const StateVec& b) -> ErrorVec {
            if constexpr (NP == NX) {
                return (a - b).eval();
            } else {
                ErrorVec e = ErrorVec::Zero();
                constexpr int min_dim = (NX < NP) ? NX : NP;
                e.template head<min_dim>() = a.template head<min_dim>() - b.template head<min_dim>();
                return e;
            }
        };
        add_x = [](const StateVec& x, const ErrorVec& dx) -> StateVec {
            if constexpr (NP == NX) {
                return (x + dx).eval();
            } else {
                StateVec res = x;
                constexpr int min_dim = (NX < NP) ? NX : NP;
                res.template head<min_dim>() += dx.template head<min_dim>();
                return res;
            }
        };
        mean_x = [](const SigmaPoints& sigs, const Eigen::Matrix<double, 2 * NP + 1, 1>& w) {
            return (sigs * w).eval();
        };

        x.setZero();
        P.setIdentity();
        Q.setIdentity();
        R.setIdentity();
    }

    bool predict(double dt, FxFunc fx, const CovMat* Q_custom = nullptr) {
        const CovMat& qm = Q_custom ? *Q_custom : Q;
        const int num_sigmas = 2 * NP + 1;

        // 1. Generate sigma points
        SigmaPoints sigmas;
        if (!generate_sigma_points(x, P, sigmas)) {
            return false;
        }

        // 2. Transform sigma points
        SigmaPoints sigmas_f;
        for (int i = 0; i < num_sigmas; ++i) {
            sigmas_f.col(i) = fx(sigmas.col(i), dt);
        }

        // 3. Predicted mean
        this->x = this->mean_x(sigmas_f, this->Wm);

        // 4. Predicted covariance
        this->P.setZero();
        for (int i = 0; i < num_sigmas; ++i) {
            ErrorVec dx = this->subtract_x(sigmas_f.col(i), this->x);
            this->P += this->Wc(i) * dx * dx.transpose();
        }
        this->P += qm * dt;
        return true;
    }

    bool update(const MeasVec& z, HxFunc hx, const Eigen::Matrix<double, NZ, NZ>* R_custom = nullptr) {
        const Eigen::Matrix<double, NZ, NZ>& rm = R_custom ? *R_custom : R;
        const int num_sigmas = 2 * NP + 1;

        // 1. Regenerate sigma points
        SigmaPoints sigmas_f;
        if (!generate_sigma_points(this->x, this->P, sigmas_f)) {
            return false;
        }

        // 2. Transform to measurement space
        Eigen::Matrix<double, NZ, 2 * NP + 1> sigmas_h;
        for (int i = 0; i < num_sigmas; ++i) {
            sigmas_h.col(i) = hx(sigmas_f.col(i));
        }

        // 3. Measurement mean and covariance
        MeasVec zp = sigmas_h * this->Wm;

        Eigen::Matrix<double, NZ, NZ> S = Eigen::Matrix<double, NZ, NZ>::Zero();
        Eigen::Matrix<double, NP, NZ> Pxz = Eigen::Matrix<double, NP, NZ>::Zero();

        for (int i = 0; i < num_sigmas; ++i) {
            MeasVec dz = sigmas_h.col(i) - zp;
            ErrorVec dx = this->subtract_x(sigmas_f.col(i), this->x);

            S += this->Wc(i) * dz * dz.transpose();
            Pxz += this->Wc(i) * dx * dz.transpose();
        }
        S += rm;

        // 4. Correction
        Eigen::Matrix<double, NP, NZ> K = Pxz * S.inverse();
        this->x = this->add_x(this->x, K * (z - zp));
        this->P = this->P - (K * S * K.transpose());
        return true;
    }

    bool generate_sigma_points(const StateVec& state, const CovMat& cov, SigmaPoints& sigmas) {
        sigmas.col(0) = state;

        CovMat P_sym = (cov + cov.transpose()) * 0.5;
        // Cholesky decomposition
        Eigen::LLT<CovMat> llt((NP + lambda) * P_sym);
        if (llt.info() != Eigen::Success) {
            last_error = UKFError::CholeskyFailed;
            return false;
        }
        CovMat L = llt.matrixL();

        for (int i = 0; i < NP; ++i) {
            ErrorVec Lcol = L.col(i);
            sigmas.col(i + 1) = add_x(state, Lcol);
            sigmas.col(i + 1 + NP) = add_x(state, -Lcol);
        }
        return true;
    }

    UKFError getLastError() const { return last_error; }

    // Parameters and State
    double alpha, beta, kappa, lambda, gamma;
    StateVec x;
    CovMat P, Q;
    Eigen::Matrix<double, NZ, NZ> R;
    Eigen::Matrix<double, 2 * NP + 1, 1> Wm, Wc;

    // Callbacks
    SubtractFunc subtract_x;
    AddFunc add_x;
    MeanFunc mean_x;

private:
    UKFError last_error;
};

} // namespace kalman_filters
} // namespace opengnc
