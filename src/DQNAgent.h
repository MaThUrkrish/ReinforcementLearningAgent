#ifndef REMAP2D_DQN_AGENT_H
#define REMAP2D_DQN_AGENT_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "MazeEnvironment.h"
#include "utils/Random.h"

namespace remap2d {

// A transition stored in the replay buffer or used for online updates.
struct Transition {
    StateVec state;
    int      action;
    float    reward;
    StateVec next_state;
    bool     done;
};

// DQN agent with a small MLP: input → hidden (ReLU) → Q-values (linear).
// Uses a target network for stable TD targets, Huber loss, and Adam optimizer.
class DQNAgent {
public:
    DQNAgent(int input_dim, int hidden_size, int num_actions,
             float lr, float gamma, float huber_delta,
             int target_update_freq, Random& rng);

    // Epsilon-greedy action selection.
    int select_action(const StateVec& state, float epsilon);

    // Train on a batch of transitions (from replay buffer or online).
    void train_step(const std::vector<Transition>& batch);

    // Convenience: train on a single transition (online / no-replay condition).
    void train_step_online(const Transition& t) { train_step({t}); }

    // Get hidden-layer activations (post-ReLU) for a state.
    std::vector<float> get_hidden_activations(const StateVec& state) const;

    // Get hidden-layer pre-ReLU activations (for analysis).
    std::vector<float> get_hidden_preact(const StateVec& state) const;

    // Get Q-values for a state.
    std::vector<float> get_q_values(const StateVec& state) const;

    // Update target network (hard copy).
    void update_target();

    int hidden_size() const { return hidden_size_; }
    int input_dim()   const { return input_dim_; }
    int num_actions() const { return num_actions_; }

    // Save/load weights to a simple binary file (for reproducibility).
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    int input_dim_;
    int hidden_size_;
    int num_actions_;
    float lr_;
    float gamma_;
    float huber_delta_;
    int  target_update_freq_;
    Random& rng_;

    // Online network weights
    Eigen::MatrixXf W1_;  // [input_dim, hidden]
    Eigen::VectorXf b1_;  // [hidden]
    Eigen::MatrixXf W2_;  // [hidden, num_actions]
    Eigen::VectorXf b2_;  // [num_actions]

    // Target network weights
    Eigen::MatrixXf W1_tgt_;
    Eigen::VectorXf b1_tgt_;
    Eigen::MatrixXf W2_tgt_;
    Eigen::VectorXf b2_tgt_;

    // Adam moments
    Eigen::MatrixXf mW1_, vW1_, mW2_, vW2_;
    Eigen::VectorXf mb1_, vb1_, mb2_, vb2_;
    long long adam_t_ = 0;
    long long step_count_ = 0;

    // Adam hyperparameters
    static constexpr float beta1_ = 0.9f;
    static constexpr float beta2_ = 0.999f;
    static constexpr float adam_eps_ = 1e-8f;

    // Forward pass: returns hidden (post-ReLU) and Q-values.
    void forward(const StateVec& state, Eigen::VectorXf& hidden,
                 Eigen::VectorXf& q_values) const;

    // Forward pass on target network
    void forward_target(const StateVec& state, Eigen::VectorXf& q_values) const;

    // Initialize weights (He initialization)
    void init_weights();

    // Adam update for a matrix parameter
    void adam_update_mat(Eigen::MatrixXf& param, const Eigen::MatrixXf& grad,
                         Eigen::MatrixXf& m, Eigen::MatrixXf& v);
    void adam_update_vec(Eigen::VectorXf& param, const Eigen::VectorXf& grad,
                         Eigen::VectorXf& m, Eigen::VectorXf& v);
};

} // namespace remap2d

#endif // REMAP2D_DQN_AGENT_H
