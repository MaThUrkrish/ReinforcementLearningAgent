#include "RunLoader.h"
#include <sqlite3.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include "Config.h"  // for basename()

namespace remap2d {

static std::string find_db_path(const std::string& run_dir) {
    // Look for run_<basename>.db in the run directory
    std::string base = basename(run_dir);
    std::string db_path = run_dir + "/run_" + base + ".db";

    std::ifstream test(db_path);
    if (test.good()) return db_path;

    // Fallback: search for any .db file in the directory
    // (not portable, but sufficient for this tool)
    std::string cmd = "ls " + run_dir + "/*.db 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[4096];
        if (fgets(buf, sizeof(buf), pipe)) {
            std::string result(buf);
            // trim newline
            while (!result.empty() && result.back() == '\n') result.pop_back();
            pclose(pipe);
            if (!result.empty()) return result;
        }
        pclose(pipe);
    }
    throw std::runtime_error("No .db file found in " + run_dir);
}

RunData load_run(const std::string& run_dir) {
    RunData data;
    data.run_dir = run_dir;

    std::string db_path = find_db_path(run_dir);

    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
        std::string msg = db ? sqlite3_errmsg(db) : "cannot open";
        if (db) sqlite3_close(db);
        throw std::runtime_error("Cannot open database " + db_path + ": " + msg);
    }

    try {
        // ── Load experiment metadata ──
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db,
                "SELECT run_id, config_hash, timestamp, replay_enabled, "
                "       hidden_size, num_contexts, seed FROM experiments LIMIT 1",
                -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Query experiments failed");
        }
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            data.meta.run_id         = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            data.meta.config_hash    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            data.meta.timestamp      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            data.meta.replay_enabled = sqlite3_column_int(stmt, 3) != 0;
            data.meta.hidden_size    = sqlite3_column_int(stmt, 4);
            data.meta.num_contexts   = sqlite3_column_int(stmt, 5);
            data.meta.seed           = sqlite3_column_int(stmt, 6);
        }
        sqlite3_finalize(stmt);

        // ── Load contexts ──
        if (sqlite3_prepare_v2(db,
                "SELECT context_id, maze_name, grid_w, grid_h FROM contexts ORDER BY context_id",
                -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Query contexts failed");
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            ContextMeta c;
            c.context_id = sqlite3_column_int(stmt, 0);
            c.maze_name  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            c.grid_w     = sqlite3_column_int(stmt, 2);
            c.grid_h     = sqlite3_column_int(stmt, 3);
            data.contexts.push_back(c);
        }
        sqlite3_finalize(stmt);

        // ── Load episodes ──
        if (sqlite3_prepare_v2(db,
                "SELECT episode_id, context_id, total_reward, steps "
                "FROM episodes ORDER BY episode_id",
                -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Query episodes failed");
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            EpisodeMeta e;
            e.episode_id   = sqlite3_column_int(stmt, 0);
            e.context_id   = sqlite3_column_int(stmt, 1);
            e.total_reward = static_cast<float>(sqlite3_column_double(stmt, 2));
            e.steps        = sqlite3_column_int(stmt, 3);
            data.episodes.push_back(e);
        }
        sqlite3_finalize(stmt);

        // ── Load activations ──
        if (sqlite3_prepare_v2(db,
                "SELECT context_id, episode_id, step, pos_x, pos_y, hidden "
                "FROM activations ORDER BY context_id, episode_id, step",
                -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Query activations failed");
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            LoadedActivation a;
            a.context_id = sqlite3_column_int(stmt, 0);
            a.episode_id = sqlite3_column_int(stmt, 1);
            a.step       = sqlite3_column_int(stmt, 2);
            a.pos_x      = sqlite3_column_int(stmt, 3);
            a.pos_y      = sqlite3_column_int(stmt, 4);

            const void* blob = sqlite3_column_blob(stmt, 5);
            int blob_bytes   = sqlite3_column_bytes(stmt, 5);
            int num_floats   = blob_bytes / sizeof(float);
            a.hidden.resize(num_floats);
            if (blob && num_floats > 0) {
                std::memcpy(a.hidden.data(), blob, blob_bytes);
            }
            data.activations.push_back(a);
        }
        sqlite3_finalize(stmt);

    } catch (...) {
        sqlite3_close(db);
        throw;
    }

    sqlite3_close(db);
    return data;
}

} // namespace remap2d
