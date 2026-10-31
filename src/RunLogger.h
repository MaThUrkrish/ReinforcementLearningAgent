#ifndef REMAP2D_RUN_LOGGER_H
#define REMAP2D_RUN_LOGGER_H

#include <string>
#include <vector>
#include "Config.h"

namespace remap2d {

// Buffers activation samples and episode summaries in memory, then
// flushes everything to a single SQLite database file on disk.
//
// SQLite schema:
//   experiments (run_id PK, config_hash, timestamp, replay_enabled,
//                hidden_size, num_contexts, seed)
//   contexts    (context_id, run_id FK, maze_name, grid_w, grid_h)
//   episodes    (episode_id, run_id FK, context_id, total_reward, steps, timestamp)
//   activations (run_id, context_id, episode_id, step, pos_x, pos_y, hidden BLOB)
//
// Activations are stored as raw float32 BLOBs (replacing the original HDF5
// design — SQLite BLOBs are simpler, sufficient at this scale, and keep the
// entire data store in one inspectable file).

struct ActivationSample {
    int episode_id;
    int context_id;
    int step;
    int pos_x;
    int pos_y;
    std::vector<float> hidden;  // [hidden_size]
};

struct EpisodeSummary {
    int episode_id;
    int context_id;
    float total_reward;
    int steps;
    std::string timestamp;
};

class RunLogger {
public:
    explicit RunLogger(const Config& cfg) : cfg_(cfg) {}

    void log_activation(int episode_id, int context_id, int step,
                        int pos_x, int pos_y, const std::vector<float>& hidden) {
        activations_.push_back({episode_id, context_id, step, pos_x, pos_y, hidden});
    }

    void log_episode(int episode_id, int context_id, float reward, int steps) {
        episodes_.push_back({episode_id, context_id, reward, steps, ""});
    }

    // Write everything to <run_dir>/run_<id>.db + <run_dir>/config.txt
    void flush_to_disk(const std::string& run_dir, const std::vector<std::string>& maze_names);

    const Config& config() const { return cfg_; }
    size_t num_activations() const { return activations_.size(); }
    size_t num_episodes() const { return episodes_.size(); }

private:
    Config cfg_;
    std::vector<ActivationSample> activations_;
    std::vector<EpisodeSummary> episodes_;
};

} // namespace remap2d

#endif // REMAP2D_RUN_LOGGER_H
