// main.cpp — Training CLI for the 2D remapping RL agent.
//
// Usage:
//   ./train_agent --contexts 0,1,0 --episodes-per-context 300
//                 --hidden-size 64 --replay on --seed 42 --out runs/run_001
//
// Trains a DQN agent on an ABA schedule of maze contexts, logging hidden-layer
// activations and episode summaries to a SQLite database.

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <sys/stat.h>

#include "Config.h"
#include "MazeEnvironment.h"
#include "DQNAgent.h"
#include "ReplayBuffer.h"
#include "RunLogger.h"
#include "utils/Random.h"

using namespace remap2d;

static void ensure_dir(const std::string& dir) {
    mkdir(dir.c_str(), 0755);
}

int main(int argc, char* argv[]) {
    Config cfg;
    try {
        cfg = parse_train_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "=== remap2d training ===\n";
    std::cout << "  run_id:               " << cfg.run_id << "\n";
    std::cout << "  out_dir:              " << cfg.out_dir << "\n";
    std::cout << "  seed:                 " << cfg.seed << "\n";
    std::cout << "  replay:               " << (cfg.replay_enabled ? "ON" : "OFF") << "\n";
    std::cout << "  hidden_size:          " << cfg.hidden_size << "\n";
    std::cout << "  episodes_per_context: " << cfg.episodes_per_context << "\n";
    std::cout << "  context_schedule:     ";
    for (size_t i = 0; i < cfg.context_schedule.size(); ++i)
        std::cout << cfg.context_schedule[i] << (i + 1 < cfg.context_schedule.size() ? " -> " : "");
    std::cout << "\n";

    ensure_dir(cfg.out_dir);

    // ── Initialise components ──
    Random rng(cfg.seed);
    MazeEnvironment env;
    DQNAgent agent(STATE_DIM, cfg.hidden_size, NUM_ACTIONS,
                   cfg.learning_rate, cfg.gamma, cfg.huber_delta,
                   cfg.target_update_freq, rng);
    ReplayBuffer buffer(cfg.buffer_capacity, cfg.replay_enabled, rng);
    RunLogger logger(cfg);

    auto t_start = std::chrono::steady_clock::now();

    // ── Training loop ──
    int global_episode = 0;
    for (size_t sched_i = 0; sched_i < cfg.context_schedule.size(); ++sched_i) {
        int context_id = cfg.context_schedule[sched_i];
        std::cout << "\n--- Context " << context_id
                  << " (" << env.maze_name(context_id) << ") ---\n";

        for (int ep = 0; ep < cfg.episodes_per_context; ++ep) {
            env.reset(context_id);
            StateVec s = env.get_state();
            float ep_reward = 0.0f;
            int t = 0;

            for (; t < cfg.max_steps; ++t) {
                float eps = cfg.epsilon_at(ep);
                int a = agent.select_action(s, eps);
                StepResult step = env.step(a);

                Transition trans{s, a, step.reward, step.next_state, step.done};

                if (buffer.enabled()) {
                    buffer.add(trans);
                    if ((int)buffer.size() >= cfg.batch_size) {
                        auto batch = buffer.sample(cfg.batch_size);
                        agent.train_step(batch);
                    }
                } else {
                    agent.train_step_online(trans);
                }

                // Log activations periodically
                if (t % cfg.log_every == 0) {
                    auto hidden = agent.get_hidden_activations(s);
                    auto pos = env.get_position();
                    logger.log_activation(global_episode, context_id, t,
                                          pos.x, pos.y, hidden);
                }

                s = step.next_state;
                ep_reward += step.reward;
                if (step.done) break;
            }

            logger.log_episode(global_episode, context_id, ep_reward, t);

            // Progress output every 50 episodes
            if ((ep + 1) % 50 == 0 || ep == 0) {
                std::cout << "  ep " << std::setw(4) << (ep + 1) << "/"
                          << cfg.episodes_per_context
                          << "  reward=" << std::fixed << std::setprecision(2)
                          << std::setw(6) << ep_reward
                          << "  steps=" << std::setw(3) << t
                          << "  eps=" << std::setprecision(3) << cfg.epsilon_at(ep)
                          << "\n";
            }
            global_episode++;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ── Flush to disk ──
    std::vector<std::string> maze_names;
    for (int c = 0; c < env.num_contexts(); ++c)
        maze_names.push_back(env.maze_name(c));

    logger.flush_to_disk(cfg.out_dir, maze_names);

    // Save agent weights for reproducibility
    agent.save(cfg.out_dir + "/weights.bin");

    std::cout << "\n=== Training complete ===\n";
    std::cout << "  elapsed:       " << std::fixed << std::setprecision(1)
              << elapsed << " s\n";
    std::cout << "  episodes:      " << logger.num_episodes() << "\n";
    std::cout << "  activations:   " << logger.num_activations() << "\n";
    std::cout << "  database:      " << cfg.out_dir << "/run_" << cfg.run_id << ".db\n";
    std::cout << "  weights:       " << cfg.out_dir << "/weights.bin\n";
    std::cout << "\nRun analysis with:\n";
    std::cout << "  ./analyze --run " << cfg.out_dir << "\n";

    return 0;
}
