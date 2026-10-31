#include "RunLogger.h"
#include <sqlite3.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <cstring>
#include <functional>

namespace remap2d {

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

static void ensure_dir(const std::string& dir) {
    mkdir(dir.c_str(), 0755);  // ignore error if exists
}

static std::string config_hash(const Config& cfg) {
    // Simple hash of the config for reproducibility tracking
    std::ostringstream oss;
    oss << cfg.hidden_size << ":" << cfg.replay_enabled << ":"
        << cfg.episodes_per_context << ":" << cfg.seed << ":"
        << cfg.gamma << ":" << cfg.learning_rate << ":"
        << cfg.buffer_capacity << ":" << cfg.batch_size;
    return oss.str();
}

static void exec_sql(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown error";
        sqlite3_free(err);
        throw std::runtime_error("SQL error: " + msg);
    }
}

template<typename F>
static void prepare_and_bind(sqlite3* db, const char* sql, F bind_fn) {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Prepare failed: " + std::string(sqlite3_errmsg(db)));
    }
    bind_fn(stmt);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::string msg = sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        throw std::runtime_error("Step failed: " + msg);
    }
    sqlite3_finalize(stmt);
}

// ── Flush to disk ───────────────────────────────────────────────────────────

void RunLogger::flush_to_disk(const std::string& run_dir,
                               const std::vector<std::string>& maze_names) {
    ensure_dir(run_dir);
    std::string db_path = run_dir + "/run_" + cfg_.run_id + ".db";

    // Remove existing file (overwrite)
    std::remove(db_path.c_str());

    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
        std::string msg = db ? sqlite3_errmsg(db) : "cannot open";
        if (db) sqlite3_close(db);
        throw std::runtime_error("Cannot open database: " + msg);
    }

    try {
        exec_sql(db, "PRAGMA journal_mode=WAL;");
        exec_sql(db, "PRAGMA synchronous=NORMAL;");

        // ── Create tables ──
        exec_sql(db,
            "CREATE TABLE experiments ("
            "  run_id TEXT PRIMARY KEY,"
            "  config_hash TEXT,"
            "  timestamp TEXT,"
            "  replay_enabled INTEGER,"
            "  hidden_size INTEGER,"
            "  num_contexts INTEGER,"
            "  seed INTEGER"
            ");");

        exec_sql(db,
            "CREATE TABLE contexts ("
            "  context_id INTEGER,"
            "  run_id TEXT REFERENCES experiments(run_id),"
            "  maze_name TEXT,"
            "  grid_w INTEGER,"
            "  grid_h INTEGER,"
            "  PRIMARY KEY (context_id, run_id)"
            ");");

        exec_sql(db,
            "CREATE TABLE episodes ("
            "  episode_id INTEGER,"
            "  run_id TEXT REFERENCES experiments(run_id),"
            "  context_id INTEGER,"
            "  total_reward REAL,"
            "  steps INTEGER,"
            "  timestamp TEXT,"
            "  PRIMARY KEY (episode_id, run_id)"
            ");");

        exec_sql(db,
            "CREATE TABLE activations ("
            "  run_id TEXT,"
            "  context_id INTEGER,"
            "  episode_id INTEGER,"
            "  step INTEGER,"
            "  pos_x INTEGER,"
            "  pos_y INTEGER,"
            "  hidden BLOB,"
            "  PRIMARY KEY (run_id, context_id, episode_id, step)"
            ");");

        exec_sql(db, "BEGIN TRANSACTION;");

        // ── Insert experiment metadata ──
        prepare_and_bind(db,
            "INSERT INTO experiments VALUES (?,?,?,?,?,?,?)",
            [&](sqlite3_stmt* stmt) {
                std::string ts = current_timestamp();
                sqlite3_bind_text(stmt, 1, cfg_.run_id.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(stmt, 2, config_hash(cfg_).c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(stmt, 3, ts.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_int(stmt, 4, cfg_.replay_enabled ? 1 : 0);
                sqlite3_bind_int(stmt, 5, cfg_.hidden_size);
                int num_ctx = 0;
                for (int c : cfg_.context_schedule)
                    num_ctx = std::max(num_ctx, c + 1);
                sqlite3_bind_int(stmt, 6, num_ctx);
                sqlite3_bind_int(stmt, 7, static_cast<int>(cfg_.seed));
            });

        // ── Insert contexts ──
        int num_ctx = 0;
        for (int c : cfg_.context_schedule)
            num_ctx = std::max(num_ctx, c + 1);
        for (int c = 0; c < num_ctx; ++c) {
            std::string name = (c < (int)maze_names.size()) ? maze_names[c] : ("maze" + std::to_string(c));
            prepare_and_bind(db,
                "INSERT INTO contexts VALUES (?,?,?,?,?)",
                [&](sqlite3_stmt* stmt) {
                    sqlite3_bind_int(stmt, 1, c);
                    sqlite3_bind_text(stmt, 2, cfg_.run_id.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(stmt, 3, name.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_int(stmt, 4, 10);  // GRID_W
                    sqlite3_bind_int(stmt, 5, 10);  // GRID_H
                });
        }

        // ── Insert episodes ──
        // Note: episode_id is unique per run, not per context. We number
        // sequentially across the whole schedule.
        sqlite3_stmt* ep_stmt = nullptr;
        if (sqlite3_prepare_v2(db,
                "INSERT INTO episodes VALUES (?,?,?,?,?,?)", -1,
                &ep_stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Prepare episodes failed");
        }
        std::string ts = current_timestamp();
        for (const auto& ep : episodes_) {
            sqlite3_bind_int(ep_stmt, 1, ep.episode_id);
            sqlite3_bind_text(ep_stmt, 2, cfg_.run_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(ep_stmt, 3, ep.context_id);
            sqlite3_bind_double(ep_stmt, 4, static_cast<double>(ep.total_reward));
            sqlite3_bind_int(ep_stmt, 5, ep.steps);
            sqlite3_bind_text(ep_stmt, 6, ts.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(ep_stmt);
            sqlite3_reset(ep_stmt);
        }
        sqlite3_finalize(ep_stmt);

        // ── Insert activations ──
        sqlite3_stmt* act_stmt = nullptr;
        if (sqlite3_prepare_v2(db,
                "INSERT INTO activations VALUES (?,?,?,?,?,?,?)", -1,
                &act_stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Prepare activations failed");
        }
        for (const auto& a : activations_) {
            sqlite3_bind_text(act_stmt, 1, cfg_.run_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(act_stmt, 2, a.context_id);
            sqlite3_bind_int(act_stmt, 3, a.episode_id);
            sqlite3_bind_int(act_stmt, 4, a.step);
            sqlite3_bind_int(act_stmt, 5, a.pos_x);
            sqlite3_bind_int(act_stmt, 6, a.pos_y);
            int blob_bytes = static_cast<int>(a.hidden.size() * sizeof(float));
            sqlite3_bind_blob(act_stmt, 7,
                a.hidden.data(), blob_bytes, SQLITE_STATIC);
            sqlite3_step(act_stmt);
            sqlite3_reset(act_stmt);
        }
        sqlite3_finalize(act_stmt);

        exec_sql(db, "COMMIT;");

        // ── Create index for faster analysis queries ──
        exec_sql(db, "CREATE INDEX idx_act_ctx ON activations(context_id);");
        exec_sql(db, "CREATE INDEX idx_ep_ctx ON episodes(context_id);");

    } catch (...) {
        sqlite3_close(db);
        throw;
    }

    sqlite3_close(db);

    // ── Write config.txt for human inspection ──
    std::ofstream cfg_file(run_dir + "/config.txt");
    cfg_file << "run_id=" << cfg_.run_id << "\n";
    cfg_file << "seed=" << cfg_.seed << "\n";
    cfg_file << "replay_enabled=" << (cfg_.replay_enabled ? "true" : "false") << "\n";
    cfg_file << "hidden_size=" << cfg_.hidden_size << "\n";
    cfg_file << "episodes_per_context=" << cfg_.episodes_per_context << "\n";
    cfg_file << "max_steps=" << cfg_.max_steps << "\n";
    cfg_file << "gamma=" << cfg_.gamma << "\n";
    cfg_file << "learning_rate=" << cfg_.learning_rate << "\n";
    cfg_file << "buffer_capacity=" << cfg_.buffer_capacity << "\n";
    cfg_file << "batch_size=" << cfg_.batch_size << "\n";
    cfg_file << "target_update_freq=" << cfg_.target_update_freq << "\n";
    cfg_file << "context_schedule=";
    for (size_t i = 0; i < cfg_.context_schedule.size(); ++i) {
        if (i > 0) cfg_file << ",";
        cfg_file << cfg_.context_schedule[i];
    }
    cfg_file << "\n";
    cfg_file << "num_activations=" << activations_.size() << "\n";
    cfg_file << "num_episodes=" << episodes_.size() << "\n";
    cfg_file.close();
}

} // namespace remap2d
