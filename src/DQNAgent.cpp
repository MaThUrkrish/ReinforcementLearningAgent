#include "DQNAgent.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace remap2d {

// ── Constructor ─────────────────────────────────────────────────────────────

DQNAgent::DQNAgent(int input_dim, int hidden_size, int num_actions,
                   float lr, float gamma, float huber_delta,
                   int target_update_freq, Random& rng)
    : input_dim_(input_dim), hidden_size_(hidden_size), num_actions_(num_actions),
      lr_(lr), gamma_(gamma), huber_delta_(huber_delta),
      target_update_freq_(target_update_freq), rng_(rng)
{
    init_weights();
}

// ── Weight initialization (He) ──────────────────────────────────────────────

void DQNAgent::init_weights() {
    float std1 = std::sqrt(2.0f / static_cast<float>(input_dim_));
    float std2 = std::sqrt(2.0f / static_cast<float>(hidden_size_));

    W1_.resize(input_dim_, hidden_size_);
    W2_.resize(hidden_size_, num_actions_);
    b1_ = Eigen::VectorXf::Zero(hidden_size_);
    b2_ = Eigen::VectorXf::Zero(num_actions_);

    for (int j = 0; j < hidden_size_; ++j)
        for (int i = 0; i < input_dim_; ++i)
            W1_(i, j) = rng_.normal() * std1;

    for (int j = 0; j < num_actions_; ++j)
        for (int i = 0; i < hidden_size_; ++i)
            W2_(i, j) = rng_.normal() * std2;

    // Target network starts as a copy
    W1_tgt_ = W1_; b1_tgt_ = b1_;
    W2_tgt_ = W2_; b2_tgt_ = b2_;

    // Adam moments → zero
    mW1_ = Eigen::MatrixXf::Zero(input_dim_, hidden_size_);
    vW1_ = Eigen::MatrixXf::Zero(input_dim_, hidden_size_);
    mW2_ = Eigen::MatrixXf::Zero(hidden_size_, num_actions_);
    vW2_ = Eigen::MatrixXf::Zero(hidden_size_, num_actions_);
    mb1_ = Eigen::VectorXf::Zero(hidden_size_);
    vb1_ = Eigen::VectorXf::Zero(hidden_size_);
    mb2_ = Eigen::VectorXf::Zero(num_actions_);
    vb2_ = Eigen::VectorXf::Zero(num_actions_);
}

// ── Forward pass (single state) ─────────────────────────────────────────────

void DQNAgent::forward(const StateVec& state,
                       Eigen::VectorXf& hidden,
                       Eigen::VectorXf& q_values) const {
    Eigen::VectorXf z1 = W1_.transpose() * state + b1_;
    hidden = z1.cwiseMax(0.0f);
    q_values = W2_.transpose() * hidden + b2_;
}

// ── Action selection (epsilon-greedy) ───────────────────────────────────────

int DQNAgent::select_action(const StateVec& state, float epsilon) {
    if (rng_.bernoulli(epsilon)) {
        return rng_.randint(0, num_actions_ - 1);
    }
    Eigen::VectorXf hidden, q;
    forward(state, hidden, q);
    int best = 0;
    q.maxCoeff(&best);
    return best;
}

// ── Training step (minibatch or online) ─────────────────────────────────────

void DQNAgent::train_step(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    int B = static_cast<int>(batch.size());

    // Build batch matrices: S [B, input], S_next [B, input]
    Eigen::MatrixXf S(B, input_dim_);
    Eigen::MatrixXf S_next(B, input_dim_);
    Eigen::VectorXf rewards(B);
    Eigen::VectorXi actions(B);
    Eigen::VectorXf dones(B);
    for (int i = 0; i < B; ++i) {
        S.row(i)      = batch[i].state.transpose();
        S_next.row(i) = batch[i].next_state.transpose();
        rewards(i)    = batch[i].reward;
        actions(i)    = batch[i].action;
        dones(i)      = batch[i].done ? 1.0f : 0.0f;
    }

    // ── Forward pass (online network) on S ──
    Eigen::MatrixXf Z1 = S * W1_;                        // [B, hidden]
    for (int i = 0; i < B; ++i) Z1.row(i) += b1_.transpose();
    Eigen::MatrixXf H  = Z1.cwiseMax(0.0f);              // [B, hidden]
    Eigen::MatrixXf Q  = H * W2_;                        // [B, num_actions]
    for (int i = 0; i < B; ++i) Q.row(i) += b2_.transpose();

    // ── Forward pass (target network) on S_next ──
    Eigen::MatrixXf Z1t = S_next * W1_tgt_;
    for (int i = 0; i < B; ++i) Z1t.row(i) += b1_tgt_.transpose();
    Eigen::MatrixXf Ht  = Z1t.cwiseMax(0.0f);
    Eigen::MatrixXf Qt  = Ht * W2_tgt_;
    for (int i = 0; i < B; ++i) Qt.row(i) += b2_tgt_.transpose();

    // ── Compute TD targets ──
    // target_i = r_i + gamma * max_a' Q_tgt(s'_i, a') * (1 - done_i)
    Eigen::VectorXf targets(B);
    for (int i = 0; i < B; ++i) {
        float max_q = Qt.row(i).maxCoeff();
        targets(i) = rewards(i) + gamma_ * max_q * (1.0f - dones(i));
    }

    // ── Compute TD errors and Huber gradients ──
    Eigen::VectorXf grad_q(B);
    for (int i = 0; i < B; ++i) {
        float q_sa = Q(i, actions(i));
        float error = q_sa - targets(i);
        float ae = std::abs(error);
        if (ae < huber_delta_) {
            grad_q(i) = error;                           // quadratic region
        } else {
            grad_q(i) = huber_delta_ * (error > 0.0f ? 1.0f : -1.0f); // linear region
        }
    }
    grad_q /= static_cast<float>(B);

    // ── dQ: scatter gradients into the taken actions ──
    Eigen::MatrixXf dQ = Eigen::MatrixXf::Zero(B, num_actions_);
    for (int i = 0; i < B; ++i)
        dQ(i, actions(i)) = grad_q(i);

    // ── Backprop ──
    Eigen::MatrixXf dW2 = H.transpose() * dQ;            // [hidden, num_actions]
    Eigen::VectorXf db2(num_actions_);
    for (int j = 0; j < num_actions_; ++j) db2(j) = dQ.col(j).sum();

    Eigen::MatrixXf dH  = dQ * W2_.transpose();          // [B, hidden]
    Eigen::MatrixXf relu_mask = (Z1.array() > 0.0f).cast<float>();
    Eigen::MatrixXf dZ1 = dH.cwiseProduct(relu_mask);    // [B, hidden]

    Eigen::MatrixXf dW1 = S.transpose() * dZ1;           // [input, hidden]
    Eigen::VectorXf db1(hidden_size_);
    for (int j = 0; j < hidden_size_; ++j) db1(j) = dZ1.col(j).sum();

    // ── Adam update ──
    adam_t_++;
    adam_update_mat(W1_, dW1, mW1_, vW1_);
    adam_update_vec(b1_, db1, mb1_, vb1_);
    adam_update_mat(W2_, dW2, mW2_, vW2_);
    adam_update_vec(b2_, db2, mb2_, vb2_);

    // ── Target network update (hard) ──
    step_count_++;
    if (target_update_freq_ > 0 && step_count_ % target_update_freq_ == 0)
        update_target();
}

void DQNAgent::update_target() {
    W1_tgt_ = W1_; b1_tgt_ = b1_;
    W2_tgt_ = W2_; b2_tgt_ = b2_;
}

// ── Adam optimizer ──────────────────────────────────────────────────────────

void DQNAgent::adam_update_mat(Eigen::MatrixXf& param, const Eigen::MatrixXf& grad,
                                Eigen::MatrixXf& m, Eigen::MatrixXf& v) {
    m = beta1_ * m + (1.0f - beta1_) * grad;
    v = beta2_ * v + (1.0f - beta2_) * grad.cwiseProduct(grad);
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(adam_t_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(adam_t_));
    Eigen::MatrixXf m_hat = m / bc1;
    Eigen::MatrixXf v_hat = v / bc2;
    Eigen::MatrixXf denom = v_hat.cwiseSqrt();
    denom.array() += adam_eps_;
    param -= lr_ * m_hat.cwiseQuotient(denom);
}

void DQNAgent::adam_update_vec(Eigen::VectorXf& param, const Eigen::VectorXf& grad,
                                Eigen::VectorXf& m, Eigen::VectorXf& v) {
    m = beta1_ * m + (1.0f - beta1_) * grad;
    v = beta2_ * v + (1.0f - beta2_) * grad.cwiseProduct(grad);
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(adam_t_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(adam_t_));
    Eigen::VectorXf m_hat = m / bc1;
    Eigen::VectorXf v_hat = v / bc2;
    Eigen::VectorXf denom = v_hat.cwiseSqrt();
    denom.array() += adam_eps_;
    param -= lr_ * m_hat.cwiseQuotient(denom);
}

// ── Hidden activations (for logging / analysis) ─────────────────────────────

std::vector<float> DQNAgent::get_hidden_activations(const StateVec& state) const {
    Eigen::VectorXf hidden, q;
    forward(state, hidden, q);
    return std::vector<float>(hidden.data(), hidden.data() + hidden.size());
}

std::vector<float> DQNAgent::get_hidden_preact(const StateVec& state) const {
    Eigen::VectorXf z1 = W1_.transpose() * state + b1_;
    return std::vector<float>(z1.data(), z1.data() + z1.size());
}

std::vector<float> DQNAgent::get_q_values(const StateVec& state) const {
    Eigen::VectorXf hidden, q;
    forward(state, hidden, q);
    return std::vector<float>(q.data(), q.data() + q.size());
}

// ── Save / Load (binary) ────────────────────────────────────────────────────

void DQNAgent::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open " + path + " for writing");
    int dims[3] = {input_dim_, hidden_size_, num_actions_};
    ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    ofs.write(reinterpret_cast<const char*>(W1_.data()), W1_.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(b1_.data()), b1_.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(W2_.data()), W2_.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(b2_.data()), b2_.size() * sizeof(float));
}

void DQNAgent::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open " + path + " for reading");
    int dims[3];
    ifs.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (dims[0] != input_dim_ || dims[1] != hidden_size_ || dims[2] != num_actions_)
        throw std::runtime_error("Dimension mismatch in " + path);
    W1_.resize(input_dim_, hidden_size_);
    b1_.resize(hidden_size_);
    W2_.resize(hidden_size_, num_actions_);
    b2_.resize(num_actions_);
    ifs.read(reinterpret_cast<char*>(W1_.data()), W1_.size() * sizeof(float));
    ifs.read(reinterpret_cast<char*>(b1_.data()), b1_.size() * sizeof(float));
    ifs.read(reinterpret_cast<char*>(W2_.data()), W2_.size() * sizeof(float));
    ifs.read(reinterpret_cast<char*>(b2_.data()), b2_.size() * sizeof(float));
    W1_tgt_ = W1_; b1_tgt_ = b1_; W2_tgt_ = W2_; b2_tgt_ = b2_;
}

} // namespace remap2d
