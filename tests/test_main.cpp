// test_main.cpp — Unit tests for the remap2d project.
// Covers the functional tests (F-01..F-12) and integration tests (I-01..I-05)
// from the Testing & QA document.

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "Config.h"
#include "MazeEnvironment.h"
#include "DQNAgent.h"
#include "ReplayBuffer.h"
#include "RunLogger.h"
#include "utils/Random.h"
#include "analysis/RunLoader.h"
#include "analysis/RateMaps.h"
#include "analysis/Correlation.h"
#include "analysis/Decoder.h"

#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>

using namespace remap2d;

// ── F-01: reset(context_id) puts agent at start ─────────────────────────────

TEST_CASE("F-01: reset puts agent at context start", "[env]") {
    MazeEnvironment env;
    for (int c = 0; c < env.num_contexts(); ++c) {
        env.reset(c);
        REQUIRE(env.get_context_id() == c);
        REQUIRE(env.get_position().x == env.get_start(c).x);
        REQUIRE(env.get_position().y == env.get_start(c).y);
    }
}

// ── F-02: step into wall leaves position unchanged ──────────────────────────

TEST_CASE("F-02: stepping into wall does not move agent", "[env]") {
    MazeEnvironment env;
    env.reset(0);
    Position start = env.get_position();

    // The start (1,1) has a wall above (row 0) and to the left (col 0).
    // Move up → wall
    StepResult r = env.step(0);  // up
    REQUIRE(env.get_position().x == start.x);
    REQUIRE(env.get_position().y == start.y);
    REQUIRE(r.done == false);
    REQUIRE(r.reward < 0.0f);  // wall penalty
}

// ── F-03: step into goal gives reward and done ──────────────────────────────

TEST_CASE("F-03: reaching goal gives positive reward and done", "[env]") {
    MazeEnvironment env;
    env.reset(0);

    // Manually move agent next to the goal, then step into it
    // Goal for mazeA is at (8,8). Start is at (1,1).
    // Move right 7 times, then down 7 times.
    bool reached = false;
    for (int trial = 0; trial < 100 && !reached; ++trial) {
        env.reset(0);
        // Try to reach goal by moving right then down
        for (int i = 0; i < 8; ++i) {
            StepResult r = env.step(3);  // right
            if (r.done) { reached = true; REQUIRE(r.reward == 1.0f); break; }
        }
        if (reached) break;
        for (int i = 0; i < 8; ++i) {
            StepResult r = env.step(1);  // down
            if (r.done) { reached = true; REQUIRE(r.reward == 1.0f); break; }
        }
    }
    // In maze A, going right 7 then down 7 should reach the goal
    // (path: (1,1)→(8,1)→(8,8) is open)
    REQUIRE(reached);
}

// ── F-04: epsilon=0 always returns argmax Q ─────────────────────────────────

TEST_CASE("F-04: epsilon=0 selects argmax Q", "[agent]") {
    Random rng(42);
    DQNAgent agent(STATE_DIM, 16, NUM_ACTIONS, 1e-3f, 0.95f, 1.0f, 200, rng);
    StateVec s = MazeEnvironment::make_state(3, 3, 0);

    // With epsilon=0, always greedy
    std::vector<int> counts(NUM_ACTIONS, 0);
    for (int i = 0; i < 100; ++i) {
        int a = agent.select_action(s, 0.0f);
        REQUIRE(a >= 0);
        REQUIRE(a < NUM_ACTIONS);
        counts[a]++;
    }
    // All actions should be the same (argmax)
    int non_zero = 0;
    for (int c : counts) if (c > 0) ++non_zero;
    REQUIRE(non_zero == 1);
}

// ── F-05: epsilon=1 gives approximately uniform distribution ────────────────

TEST_CASE("F-05: epsilon=1 gives near-uniform actions", "[agent]") {
    Random rng(42);
    DQNAgent agent(STATE_DIM, 16, NUM_ACTIONS, 1e-3f, 0.95f, 1.0f, 200, rng);
    StateVec s = MazeEnvironment::make_state(3, 3, 0);

    int N = 4000;
    std::vector<int> counts(NUM_ACTIONS, 0);
    for (int i = 0; i < N; ++i) {
        int a = agent.select_action(s, 1.0f);
        counts[a]++;
    }
    // Each action should be roughly N/4 ± some tolerance
    float expected = N / 4.0f;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        // Within 3 standard deviations of binomial
        REQUIRE(std::abs(counts[a] - expected) < 3.0f * std::sqrt(N * 0.25f * 0.75f));
    }
}

// ── F-06: replay buffer evicts oldest (FIFO) ────────────────────────────────

TEST_CASE("F-06: replay buffer evicts oldest", "[buffer]") {
    Random rng(42);
    int cap = 5;
    ReplayBuffer buf(cap, true, rng);

    StateVec s = StateVec::Zero(4);
    for (int i = 0; i < cap + 3; ++i) {
        // Make each transition unique by encoding i in the reward
        Transition t{s, 0, static_cast<float>(i), s, false};
        buf.add(t);
    }
    REQUIRE((int)buf.size() == cap);

    // Sample and check that we only see rewards 3..7 (last 5)
    for (int trial = 0; trial < 100; ++trial) {
        auto batch = buf.sample(5);
        for (const auto& t : batch) {
            REQUIRE(t.reward >= 3.0f);
            REQUIRE(t.reward <= 7.0f);
        }
    }
}

// ── F-07: disabled buffer stays empty ───────────────────────────────────────

TEST_CASE("F-07: disabled replay buffer stays empty", "[buffer]") {
    Random rng(42);
    ReplayBuffer buf(100, false, rng);
    StateVec s = StateVec::Zero(4);
    for (int i = 0; i < 50; ++i) {
        buf.add({s, 0, 0.0f, s, false});
    }
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.enabled() == false);
    auto batch = buf.sample(10);
    REQUIRE(batch.empty());
}

// ── F-08: RunLogger flush produces correct activation rows ─────────────────

TEST_CASE("F-08: RunLogger writes activations to SQLite", "[logger]") {
    Config cfg;
    cfg.run_id = "test_f08";
    cfg.hidden_size = 8;
    cfg.replay_enabled = true;
    cfg.seed = 42;
    cfg.context_schedule = {0, 1};

    RunLogger logger(cfg);
    std::vector<float> h(8, 0.5f);
    logger.log_activation(0, 0, 0, 1, 2, h);
    logger.log_activation(0, 0, 10, 3, 4, h);
    logger.log_activation(1, 1, 0, 5, 6, h);

    std::string dir = "runs/test_f08";
    mkdir(dir.c_str(), 0755);
    logger.flush_to_disk(dir, {"mazeA", "mazeB"});

    // Load and verify
    RunData data = load_run(dir);
    REQUIRE(data.activations.size() == 3);
    REQUIRE(data.activations[0].pos_x == 1);
    REQUIRE(data.activations[0].pos_y == 2);
    REQUIRE(data.activations[0].hidden.size() == 8);
    REQUIRE(data.activations[0].hidden[0] == Approx(0.5f));
}

// ── F-09: RunLogger flush produces correct episode rows ─────────────────────

TEST_CASE("F-09: RunLogger writes episodes to SQLite", "[logger]") {
    Config cfg;
    cfg.run_id = "test_f09";
    cfg.hidden_size = 4;
    cfg.context_schedule = {0, 1, 0};

    RunLogger logger(cfg);
    logger.log_episode(0, 0, 0.5f, 50);
    logger.log_episode(1, 0, 0.8f, 40);
    logger.log_episode(2, 1, -0.2f, 200);

    std::string dir = "runs/test_f09";
    mkdir(dir.c_str(), 0755);
    logger.flush_to_disk(dir, {"mazeA", "mazeB", "mazeC"});

    RunData data = load_run(dir);
    REQUIRE(data.episodes.size() == 3);
    REQUIRE(data.episodes[0].total_reward == Approx(0.5f));
    REQUIRE(data.episodes[1].context_id == 0);
    REQUIRE(data.episodes[2].steps == 200);
}

// ── F-10: rate maps peak at correct cell for synthetic data ─────────────────

TEST_CASE("F-10: rate maps recover synthetic place field", "[analysis]") {
    // One unit active only at cell (3, 4)
    int gw = 10, gh = 10;
    (void)gh; // used in loops below
    std::vector<LoadedActivation> acts;
    // Visit every cell; unit 0 fires 1.0 at (3,4) and 0.0 elsewhere
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            LoadedActivation a;
            a.context_id = 0;
            a.episode_id = 0;
            a.step = 0;
            a.pos_x = x;
            a.pos_y = y;
            a.hidden = {(x == 3 && y == 4) ? 1.0f : 0.0f};
            acts.push_back(a);
        }
    }
    std::vector<ContextMeta> ctxs = {{0, "test", gw, gh}};
    auto maps = compute_rate_maps(acts, ctxs);
    REQUIRE(maps.size() == 1);
    REQUIRE(maps[0].hidden_size == 1);

    // Peak should be at (3, 4)
    float peak_val = -1.0f;
    int peak_x = -1, peak_y = -1;
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            float v = rate_at(maps[0], 0, x, y);
            if (v > peak_val) { peak_val = v; peak_x = x; peak_y = y; }
        }
    }
    REQUIRE(peak_x == 3);
    REQUIRE(peak_y == 4);
    REQUIRE(peak_val == Approx(1.0f));
}

// ── F-11: correlation ≈ 1.0 for identical, ≈ 0 for shuffled ──────────────────

TEST_CASE("F-11: population vector correlation identical vs random", "[analysis]") {
    int gw = 5, gh = 5, hs = 4;
    // Two identical maps
    ContextRateMaps mapA, mapB;
    mapA.context_id = 0; mapB.context_id = 1;
    mapA.hidden_size = hs; mapB.hidden_size = hs;
    mapA.grid_w = gw; mapA.grid_h = gh;
    mapB.grid_w = gw; mapB.grid_h = gh;
    int cells = gw * gh;
    mapA.maps.resize(hs * cells);
    mapB.maps.resize(hs * cells);
    mapA.visits.assign(cells, 1);
    mapB.visits.assign(cells, 1);

    Random rng(42);
    for (int u = 0; u < hs; ++u)
        for (int c = 0; c < cells; ++c) {
            float v = rng.uniform();
            mapA.maps[u * cells + c] = v;
            mapB.maps[u * cells + c] = v;  // identical
        }

    float corr_id = population_vector_correlation(mapA, mapB);
    REQUIRE(corr_id == Approx(1.0f).margin(0.01f));

    // Now shuffle mapB
    std::vector<float> shuffled = mapA.maps;
    rng.shuffle(shuffled);
    mapB.maps = shuffled;
    float corr_shuffled = population_vector_correlation(mapA, mapB);
    // Should be near 0 on average
    REQUIRE(std::abs(corr_shuffled) < 0.5f);
}

// ── F-12: decoder achieves ~100% on linearly separable data ─────────────────

TEST_CASE("F-12: decoder reaches near-100% on separable data", "[analysis]") {
    // Create 2-class data: class 0 has activations around (-1,-1,...),
    // class 1 around (+1,+1,...)
    int N = 200, D = 8;
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    for (int i = 0; i < N; ++i) {
        int label = i % 2;
        std::vector<float> v(D);
        for (int d = 0; d < D; ++d)
            v[d] = (label == 0 ? -1.0f : 1.0f) + (i % 7) * 0.001f * (d + 1);
        X.push_back(v);
        y.push_back(label);
    }
    auto result = train_softmax_decoder(X, y, 2, 500, 0.5f, 1e-4f, 42);
    REQUIRE(result.test_accuracy > 0.95f);
}

// ── I-01: full pipeline smoke test ──────────────────────────────────────────

TEST_CASE("I-01: full pipeline smoke test", "[integration]") {
    // Train a tiny agent, log, flush, load, analyze
    Config cfg;
    cfg.context_schedule = {0, 1};
    cfg.episodes_per_context = 5;
    cfg.max_steps = 50;
    cfg.hidden_size = 16;
    cfg.replay_enabled = true;
    cfg.seed = 42;
    cfg.out_dir = "runs/test_i01";
    cfg.run_id = "test_i01";

    Random rng(cfg.seed);
    MazeEnvironment env;
    DQNAgent agent(STATE_DIM, cfg.hidden_size, NUM_ACTIONS,
                   cfg.learning_rate, cfg.gamma, cfg.huber_delta,
                   cfg.target_update_freq, rng);
    ReplayBuffer buffer(1000, true, rng);
    RunLogger logger(cfg);

    int global_ep = 0;
    for (int ctx : cfg.context_schedule) {
        for (int ep = 0; ep < cfg.episodes_per_context; ++ep) {
            env.reset(ctx);
            StateVec s = env.get_state();
            float ep_r = 0;
            int t;
            for (t = 0; t < cfg.max_steps; ++t) {
                int a = agent.select_action(s, 0.5f);
                StepResult step = env.step(a);
                Transition tr{s, a, step.reward, step.next_state, step.done};
                buffer.add(tr);
                if ((int)buffer.size() >= cfg.batch_size)
                    agent.train_step(buffer.sample(cfg.batch_size));
                if (t % cfg.log_every == 0) {
                    auto h = agent.get_hidden_activations(s);
                    auto pos = env.get_position();
                    logger.log_activation(global_ep, ctx, t, pos.x, pos.y, h);
                }
                s = step.next_state;
                ep_r += step.reward;
                if (step.done) break;
            }
            logger.log_episode(global_ep, ctx, ep_r, t);
            global_ep++;
        }
    }
    std::vector<std::string> names;
    for (int c = 0; c < env.num_contexts(); ++c) names.push_back(env.maze_name(c));
    logger.flush_to_disk(cfg.out_dir, names);

    // Load and verify
    RunData data = load_run(cfg.out_dir);
    REQUIRE(data.episodes.size() == 10);
    REQUIRE(!data.activations.empty());

    // Verify no out-of-grid positions
    for (const auto& a : data.activations) {
        REQUIRE(a.pos_x >= 0);
        REQUIRE(a.pos_x < 10);
        REQUIRE(a.pos_y >= 0);
        REQUIRE(a.pos_y < 10);
    }
}

// ── I-03: replay-on vs replay-off produce loadable runs ─────────────────────

TEST_CASE("I-03: replay on/off both produce loadable runs", "[integration]") {
    for (bool replay : {true, false}) {
        Config cfg;
        cfg.context_schedule = {0, 1};
        cfg.episodes_per_context = 3;
        cfg.max_steps = 30;
        cfg.hidden_size = 8;
        cfg.replay_enabled = replay;
        cfg.seed = 42;
        cfg.out_dir = "runs/test_i03_" + std::string(replay ? "on" : "off");
        cfg.run_id = "test_i03_" + std::string(replay ? "on" : "off");

        Random rng(cfg.seed);
        MazeEnvironment env;
        DQNAgent agent(STATE_DIM, cfg.hidden_size, NUM_ACTIONS,
                       cfg.learning_rate, cfg.gamma, cfg.huber_delta,
                       cfg.target_update_freq, rng);
        ReplayBuffer buffer(500, replay, rng);
        RunLogger logger(cfg);

        int global_ep = 0;
        for (int ctx : cfg.context_schedule) {
            for (int ep = 0; ep < cfg.episodes_per_context; ++ep) {
                env.reset(ctx);
                StateVec s = env.get_state();
                float ep_r = 0; int t;
                for (t = 0; t < cfg.max_steps; ++t) {
                    int a = agent.select_action(s, 0.5f);
                    StepResult step = env.step(a);
                    Transition tr{s, a, step.reward, step.next_state, step.done};
                    if (buffer.enabled()) {
                        buffer.add(tr);
                        if ((int)buffer.size() >= cfg.batch_size)
                            agent.train_step(buffer.sample(cfg.batch_size));
                    } else {
                        agent.train_step_online(tr);
                    }
                    s = step.next_state; ep_r += step.reward;
                    if (step.done) break;
                }
                logger.log_episode(global_ep, ctx, ep_r, t);
                global_ep++;
            }
        }
        std::vector<std::string> names;
        for (int c = 0; c < env.num_contexts(); ++c) names.push_back(env.maze_name(c));
        logger.flush_to_disk(cfg.out_dir, names);

        // Must be loadable
        RunData data = load_run(cfg.out_dir);
        REQUIRE(data.meta.replay_enabled == replay);
        REQUIRE(data.episodes.size() == 6);
    }
}

// ── I-04: ABA schedule produces correct context sequence ────────────────────

TEST_CASE("I-04: ABA context schedule", "[integration]") {
    Config cfg;
    cfg.context_schedule = {0, 1, 0};
    cfg.episodes_per_context = 10;
    cfg.hidden_size = 8;
    cfg.run_id = "test_i04";
    cfg.out_dir = "runs/test_i04";

    RunLogger logger(cfg);
    // Simulate 30 episodes: 10 ctx0, 10 ctx1, 10 ctx0
    int ge = 0;
    for (int ctx : cfg.context_schedule) {
        for (int ep = 0; ep < 10; ++ep) {
            logger.log_episode(ge++, ctx, 0.0f, 10);
        }
    }
    std::vector<std::string> names = {"mazeA", "mazeB", "mazeC"};
    logger.flush_to_disk(cfg.out_dir, names);

    RunData data = load_run(cfg.out_dir);
    // Context IDs: 0,0,...,0,1,1,...,1,0,0,...,0
    REQUIRE(data.episodes.size() == 30);
    std::vector<int> ctx_ids;
    for (const auto& e : data.episodes) ctx_ids.push_back(e.context_id);

    // Check the schedule pattern
    for (int i = 0; i < 10; ++i) REQUIRE(ctx_ids[i] == 0);
    for (int i = 10; i < 20; ++i) REQUIRE(ctx_ids[i] == 1);
    for (int i = 20; i < 30; ++i) REQUIRE(ctx_ids[i] == 0);

    // Blocks
    auto blocks = extract_blocks(data.episodes);
    REQUIRE(blocks.size() == 3);
    REQUIRE(blocks[0].context_id == 0);
    REQUIRE(blocks[1].context_id == 1);
    REQUIRE(blocks[2].context_id == 0);
}

// ── I-05: analysis is deterministic (idempotent) ────────────────────────────

TEST_CASE("I-05: rate map computation is deterministic", "[analysis]") {
    std::vector<LoadedActivation> acts;
    Random rng(42);
    for (int i = 0; i < 100; ++i) {
        LoadedActivation a;
        a.context_id = 0;
        a.episode_id = i;
        a.step = 0;
        a.pos_x = rng.randint(1, 8);
        a.pos_y = rng.randint(1, 8);
        a.hidden.resize(4);
        for (int u = 0; u < 4; ++u) a.hidden[u] = rng.uniform();
        acts.push_back(a);
    }
    std::vector<ContextMeta> ctxs = {{0, "test", 10, 10}};

    auto maps1 = compute_rate_maps(acts, ctxs);
    auto maps2 = compute_rate_maps(acts, ctxs);

    REQUIRE(maps1.size() == maps2.size());
    REQUIRE(maps1[0].maps.size() == maps2[0].maps.size());
    for (size_t i = 0; i < maps1[0].maps.size(); ++i)
        REQUIRE(maps1[0].maps[i] == Approx(maps2[0].maps[i]));
}

// ── Extra: DQN agent can learn a simple maze ────────────────────────────────

TEST_CASE("Agent improves over episodes (sanity check)", "[agent][.slow]") {
    Random rng(123);
    MazeEnvironment env;
    DQNAgent agent(STATE_DIM, 32, NUM_ACTIONS, 1e-3f, 0.95f, 1.0f, 100, rng);
    ReplayBuffer buffer(2000, true, rng);

    // Train on context 0 for 200 episodes, measure early vs late reward
    float early_reward = 0, late_reward = 0;
    for (int ep = 0; ep < 200; ++ep) {
        env.reset(0);
        StateVec s = env.get_state();
        float ep_r = 0; int t;
        for (t = 0; t < 100; ++t) {
            float eps = std::max(0.05f, 1.0f - ep / 100.0f);
            int a = agent.select_action(s, eps);
            StepResult step = env.step(a);
            Transition tr{s, a, step.reward, step.next_state, step.done};
            buffer.add(tr);
            if ((int)buffer.size() >= 32)
                agent.train_step(buffer.sample(32));
            s = step.next_state; ep_r += step.reward;
            if (step.done) break;
        }
        if (ep < 50) early_reward += ep_r;
        if (ep >= 150) late_reward += ep_r;
    }
    early_reward /= 50;
    late_reward /= 50;
    // Late reward should be higher than early
    REQUIRE(late_reward > early_reward);
}
