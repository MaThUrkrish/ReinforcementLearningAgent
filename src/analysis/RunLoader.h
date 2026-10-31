#ifndef REMAP2D_RUN_LOADER_H
#define REMAP2D_RUN_LOADER_H

#include <string>
#include <vector>
#include <map>

namespace remap2d {

// Run metadata loaded from the experiments table.
struct RunMeta {
    std::string run_id;
    std::string config_hash;
    std::string timestamp;
    bool   replay_enabled;
    int    hidden_size;
    int    num_contexts;
    int    seed;
};

// Context metadata.
struct ContextMeta {
    int context_id;
    std::string maze_name;
    int grid_w;
    int grid_h;
};

// Episode metadata.
struct EpisodeMeta {
    int   episode_id;
    int   context_id;
    float total_reward;
    int   steps;
};

// Activation sample loaded from the activations table.
struct LoadedActivation {
    int context_id;
    int episode_id;
    int step;
    int pos_x;
    int pos_y;
    std::vector<float> hidden;
};

// All data loaded from a run directory.
struct RunData {
    RunMeta meta;
    std::vector<ContextMeta> contexts;
    std::vector<EpisodeMeta> episodes;
    std::vector<LoadedActivation> activations;
    std::string run_dir;
};

// Load a run from <run_dir>/run_<basename>.db
// Also reads <run_dir>/config.txt for display.
RunData load_run(const std::string& run_dir);

} // namespace remap2d

#endif // REMAP2D_RUN_LOADER_H
