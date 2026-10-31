#include "MazeEnvironment.h"
#include <queue>
#include <stdexcept>
#include <cassert>

namespace remap2d {

// ── Maze layouts ────────────────────────────────────────────────────────────
// Three distinct 10x10 mazes. '#' = wall, '.' = open, 'S' = start, 'G' = goal.
// All share the same grid size, start at top-left interior, goal at bottom-right.

static const std::vector<std::string> MAZE_A = {
    "##########",
    "#S.......#",
    "#........#",
    "#..####..#",
    "#..#..#..#",
    "#..#..#..#",
    "#..####..#",
    "#........#",
    "#.......G#",
    "##########",
};

static const std::vector<std::string> MAZE_B = {
    "##########",
    "#S.......#",
    "#.######.#",
    "#.#......#",
    "#.#.####.#",
    "#.#.#....#",
    "#.#.#.##.#",
    "#.#......#",
    "#.......G#",
    "##########",
};

static const std::vector<std::string> MAZE_C = {
    "##########",
    "#S...#...#",
    "#....#...#",
    "#....#...#",
    "#........#",   // gap in the vertical divider at row 4
    "#....#...#",
    "#....#...#",
    "#....#...#",
    "#....#..G#",
    "##########",
};

// ── Helpers ─────────────────────────────────────────────────────────────────

static Position find_char(const std::vector<std::string>& grid, char ch) {
    for (int y = 0; y < static_cast<int>(grid.size()); ++y)
        for (int x = 0; x < static_cast<int>(grid[y].size()); ++x)
            if (grid[y][x] == ch) return {x, y};
    return {-1, -1};
}

// ── Constructor ─────────────────────────────────────────────────────────────

MazeEnvironment::MazeEnvironment() {
    layouts_ = {MAZE_A, MAZE_B, MAZE_C};
    names_   = {"mazeA", "mazeB", "mazeC"};
    current_state_ = StateVec::Zero(STATE_DIM);

    for (int c = 0; c < static_cast<int>(layouts_.size()); ++c) {
        starts_.push_back(find_char(layouts_[c], 'S'));
        goals_.push_back(find_char(layouts_[c], 'G'));
        verify_solvable(c);
    }
}

// ── Public interface ────────────────────────────────────────────────────────

void MazeEnvironment::reset(int context_id) {
    assert(context_id >= 0 && context_id < num_contexts());
    current_context_ = context_id;
    current_pos_ = starts_[context_id];
    current_state_ = make_state(current_pos_.x, current_pos_.y, context_id);
}

StepResult MazeEnvironment::step(int action) {
    // 0=up, 1=down, 2=left, 3=right
    int dx = 0, dy = 0;
    switch (action) {
        case 0: dy = -1; break;  // up
        case 1: dy =  1; break;  // down
        case 2: dx = -1; break;  // left
        case 3: dx =  1; break;  // right
        default: assert(false && "invalid action");
    }

    int nx = current_pos_.x + dx;
    int ny = current_pos_.y + dy;

    StepResult result;
    result.done = false;

    // Check bounds and walls
    if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H ||
        layouts_[current_context_][ny][nx] == '#') {
        // Wall: stay in place, small penalty
        result.next_state = current_state_;
        result.reward = -0.02f;
        return result;
    }

    current_pos_ = {nx, ny};
    result.next_state = make_state(nx, ny, current_context_);

    // Check goal
    if (nx == goals_[current_context_].x && ny == goals_[current_context_].y) {
        result.reward = 1.0f;
        result.done = true;
    } else {
        result.reward = -0.01f;  // step penalty
    }

    current_state_ = result.next_state;
    return result;
}

bool MazeEnvironment::is_wall(int context_id, int x, int y) const {
    if (x < 0 || x >= GRID_W || y < 0 || y >= GRID_H) return true;
    return layouts_[context_id][y][x] == '#';
}

std::string MazeEnvironment::maze_name(int context_id) const {
    assert(context_id >= 0 && context_id < num_contexts());
    return names_[context_id];
}

StateVec MazeEnvironment::one_hot(int x, int y) {
    StateVec v = StateVec::Zero(POS_DIM);
    v(pos_to_index(x, y)) = 1.0f;
    return v;
}

StateVec MazeEnvironment::make_state(int x, int y, int context_id) {
    StateVec v = StateVec::Zero(STATE_DIM);
    v(pos_to_index(x, y)) = 1.0f;
    if (context_id >= 0 && context_id < MAX_CONTEXTS)
        v(POS_DIM + context_id) = 1.0f;
    return v;
}

// ── Solvability check (BFS) ─────────────────────────────────────────────────

void MazeEnvironment::verify_solvable(int context_id) const {
    const auto& grid = layouts_[context_id];
    Position start = starts_[context_id];
    Position goal  = goals_[context_id];

    std::vector<std::vector<bool>> visited(GRID_H, std::vector<bool>(GRID_W, false));
    std::queue<Position> q;
    q.push(start);
    visited[start.y][start.x] = true;

    static const int dx[] = {0, 0, -1, 1};
    static const int dy[] = {-1, 1, 0, 0};

    while (!q.empty()) {
        Position p = q.front(); q.pop();
        if (p.x == goal.x && p.y == goal.y) return;  // reachable
        for (int d = 0; d < 4; ++d) {
            int nx = p.x + dx[d], ny = p.y + dy[d];
            if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H &&
                grid[ny][nx] != '#' && !visited[ny][nx]) {
                visited[ny][nx] = true;
                q.push({nx, ny});
            }
        }
    }
    throw std::runtime_error("Maze " + names_[context_id] + " is unsolvable!");
}

} // namespace remap2d
