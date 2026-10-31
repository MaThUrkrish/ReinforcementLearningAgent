#ifndef REMAP2D_MAZE_ENVIRONMENT_H
#define REMAP2D_MAZE_ENVIRONMENT_H

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace remap2d {

// Grid-based maze environment supporting multiple contexts (layouts).
//
// The grid is GRID_W x GRID_H (including border walls).
// State = [one-hot position (GRID_W*GRID_H)] + [one-hot context (MAX_CONTEXTS)]
// The context cue is analogous to the visual cues that distinguish
// environments in the original 3D hippocampal remapping study — without
// it, the agent cannot tell which maze it is in and cannot remap.
//
// Actions: 0 = up, 1 = down, 2 = left, 3 = right.
// Moving into a wall leaves the agent in place (with a small penalty).
// Reaching the goal ends the episode with a positive reward.

static constexpr int GRID_W = 10;
static constexpr int GRID_H = 10;
static constexpr int MAX_CONTEXTS = 4;   // max number of maze layouts
static constexpr int NUM_ACTIONS = 4;
static constexpr int POS_DIM = GRID_W * GRID_H;
static constexpr int STATE_DIM = POS_DIM + MAX_CONTEXTS;

using StateVec = Eigen::VectorXf;   // one-hot, size STATE_DIM

struct Position {
    int x;  // column
    int y;  // row
};

struct StepResult {
    StateVec next_state;
    float    reward;
    bool     done;
};

class MazeEnvironment {
public:
    MazeEnvironment();

    // Number of available maze layouts.
    int num_contexts() const { return static_cast<int>(layouts_.size()); }

    // Reset to the start position of the given context.
    void reset(int context_id);

    // Take an action, return the resulting state, reward, and done flag.
    StepResult step(int action);

    // Current state as a one-hot vector.
    const StateVec& get_state() const { return current_state_; }

    int get_context_id() const { return current_context_; }
    Position get_position() const { return current_pos_; }
    Position get_start(int context_id) const { return starts_[context_id]; }
    Position get_goal(int context_id) const { return goals_[context_id]; }

    // Is (x,y) a wall in context c?
    bool is_wall(int context_id, int x, int y) const;

    // Grid dimensions
    int grid_w() const { return GRID_W; }
    int grid_h() const { return GRID_H; }

    // Maze name for logging
    std::string maze_name(int context_id) const;

    // Convert (x,y) to one-hot index
    static int pos_to_index(int x, int y) { return y * GRID_W + x; }

    // Convert one-hot index to (x,y)
    static Position index_to_pos(int idx) { return {idx % GRID_W, idx / GRID_W}; }

    // Build a state vector for a position in a given context.
    // [0, POS_DIM): one-hot position
    // [POS_DIM, POS_DIM + MAX_CONTEXTS): one-hot context
    static StateVec make_state(int x, int y, int context_id);

    // Build a position-only one-hot vector (for testing / backward compat)
    static StateVec one_hot(int x, int y);

private:
    // Maze layouts: each is GRID_H strings of GRID_W characters.
    // '#' = wall, '.' = open, 'S' = start, 'G' = goal
    std::vector<std::vector<std::string>> layouts_;
    std::vector<Position> starts_;
    std::vector<Position> goals_;
    std::vector<std::string> names_;

    int current_context_ = 0;
    Position current_pos_{0, 0};
    StateVec current_state_;

    // Verify there is a path from start to goal (BFS). Throws if unsolvable.
    void verify_solvable(int context_id) const;
};

} // namespace remap2d

#endif // REMAP2D_MAZE_ENVIRONMENT_H
