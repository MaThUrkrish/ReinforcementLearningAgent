#ifndef REMAP2D_CONFIG_H
#define REMAP2D_CONFIG_H

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace remap2d {

// ── Configuration ──────────────────────────────────────────────────────────

struct Config {
    // Experiment design
    std::vector<int> context_schedule = {0, 1, 0};  // ABA by default
    int  episodes_per_context = 300;
    int  max_steps            = 200;

    // Agent / network
    int  hidden_size       = 64;
    bool replay_enabled    = true;
    int  buffer_capacity   = 10000;
    int  batch_size        = 32;
    float gamma            = 0.95f;
    float learning_rate    = 1e-3f;
    float epsilon_start    = 1.0f;
    float epsilon_end      = 0.05f;
    float epsilon_decay_frac = 0.5f;  // decay over first 50% of episodes per context
    int   target_update_freq = 200;   // steps
    float huber_delta      = 1.0f;

    // Logging
    int  log_every = 10;  // log activations every N steps

    // Reproducibility
    unsigned int seed = 42;

    // Output
    std::string out_dir = "runs/run_001";
    std::string run_id  = "";  // empty → derived from out_dir basename

    // Derived: compute epsilon for a given episode within a context block
    float epsilon_at(int episode_in_context) const {
        int decay_episodes = static_cast<int>(episodes_per_context * epsilon_decay_frac);
        if (decay_episodes <= 0) return epsilon_end;
        float frac = static_cast<float>(episode_in_context) / decay_episodes;
        if (frac > 1.0f) frac = 1.0f;
        return epsilon_start + (epsilon_end - epsilon_start) * frac;
    }
};

// ── CLI helpers ─────────────────────────────────────────────────────────────

inline std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty())
            result.push_back(std::stoi(token));
    }
    return result;
}

inline std::string basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// Parse training CLI arguments into a Config.
// Throws std::runtime_error on malformed input.
inline Config parse_train_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if      (arg == "--contexts")             cfg.context_schedule = parse_int_list(next());
        else if (arg == "--episodes-per-context") cfg.episodes_per_context = std::stoi(next());
        else if (arg == "--max-steps")            cfg.max_steps = std::stoi(next());
        else if (arg == "--hidden-size")          cfg.hidden_size = std::stoi(next());
        else if (arg == "--replay") {
            std::string val = next();
            cfg.replay_enabled = (val == "on" || val == "1" || val == "true");
        }
        else if (arg == "--buffer-capacity")      cfg.buffer_capacity = std::stoi(next());
        else if (arg == "--batch-size")           cfg.batch_size = std::stoi(next());
        else if (arg == "--gamma")                cfg.gamma = std::stof(next());
        else if (arg == "--lr")                   cfg.learning_rate = std::stof(next());
        else if (arg == "--target-update-freq")   cfg.target_update_freq = std::stoi(next());
        else if (arg == "--log-every")            cfg.log_every = std::stoi(next());
        else if (arg == "--seed")                 cfg.seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--out")                  cfg.out_dir = next();
        else if (arg == "--run-id")               cfg.run_id = next();
        else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("Usage: train_agent --contexts 0,1,0 "
                "--episodes-per-context 300 --hidden-size 64 --replay on "
                "--seed 42 --out runs/run_001");
        }
        else
            throw std::runtime_error("Unknown argument: " + arg);
    }
    if (cfg.run_id.empty())
        cfg.run_id = basename(cfg.out_dir);
    return cfg;
}

// Parse analysis CLI arguments.
struct AnalyzeArgs {
    std::string run_dir;
    bool make_plots = true;
};

inline AnalyzeArgs parse_analyze_args(int argc, char* argv[]) {
    AnalyzeArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if      (arg == "--run")        args.run_dir = next();
        else if (arg == "--make-plots") args.make_plots = true;
        else if (arg == "--help" || arg == "-h")
            throw std::runtime_error("Usage: analyze --run runs/run_001");
        else
            throw std::runtime_error("Unknown argument: " + arg);
    }
    if (args.run_dir.empty())
        throw std::runtime_error("--run is required");
    return args;
}

} // namespace remap2d

#endif // REMAP2D_CONFIG_H
