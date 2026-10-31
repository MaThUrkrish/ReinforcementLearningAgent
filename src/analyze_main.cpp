// analyze_main.cpp — Analysis CLI for the 2D remapping RL agent.
//
// Usage:
//   ./analyze --run runs/run_001
//
// Loads a run from the SQLite database, computes rate maps, cross-context
// correlations, and linear decoder accuracies, and writes an HTML report
// with embedded SVG visualizations to <run_dir>/report.html.

#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>

#include "Config.h"
#include "analysis/RunLoader.h"
#include "analysis/RateMaps.h"
#include "analysis/Correlation.h"
#include "analysis/Decoder.h"
#include "analysis/Report.h"

using namespace remap2d;

int main(int argc, char* argv[]) {
    AnalyzeArgs args;
    try {
        args = parse_analyze_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "=== remap2d analysis ===\n";
    std::cout << "  run_dir: " << args.run_dir << "\n";

    auto t_start = std::chrono::steady_clock::now();

    // ── Load run data ──
    std::cout << "  Loading run data...\n";
    RunData data;
    try {
        data = load_run(args.run_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error loading run: " << e.what() << "\n";
        return 1;
    }

    std::cout << "  run_id:       " << data.meta.run_id << "\n";
    std::cout << "  replay:       " << (data.meta.replay_enabled ? "ON" : "OFF") << "\n";
    std::cout << "  hidden_size:  " << data.meta.hidden_size << "\n";
    std::cout << "  episodes:     " << data.episodes.size() << "\n";
    std::cout << "  activations:  " << data.activations.size() << "\n";
    std::cout << "  contexts:     " << data.contexts.size() << "\n";

    if (data.activations.empty()) {
        std::cerr << "Error: no activation data in this run.\n";
        return 1;
    }

    // ── Compute analysis ──
    std::cout << "\n  Computing rate maps...\n";
    auto blocks = extract_blocks(data.episodes);
    std::cout << "  blocks: " << blocks.size() << "\n";
    for (size_t i = 0; i < blocks.size(); ++i) {
        std::cout << "    block " << i << ": context " << blocks[i].context_id
                  << " (episodes " << blocks[i].start_episode
                  << "-" << blocks[i].end_episode << ")\n";
    }

    int hs = data.meta.hidden_size;
    int gw = data.contexts.empty() ? 10 : data.contexts[0].grid_w;
    int gh = data.contexts.empty() ? 10 : data.contexts[0].grid_h;

    std::vector<ContextRateMaps> block_maps;
    for (const auto& b : blocks) {
        block_maps.push_back(compute_block_rate_maps(data.activations, b, hs, gw, gh));
    }

    std::cout << "  Computing correlations...\n";
    auto corr_matrix = compute_correlation_matrix(block_maps);
    auto summary = analyze_remapping(blocks, corr_matrix);

    std::cout << "\n  === Correlation Matrix ===\n";
    for (size_t i = 0; i < corr_matrix.size(); ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < corr_matrix[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(3) << std::setw(7) << corr_matrix[i][j];
        }
        std::cout << "\n";
    }

    std::cout << "\n  === Remapping Summary ===\n";
    std::cout << "  avg cross-context corr: " << std::fixed << std::setprecision(3)
              << summary.avg_cross_context_corr << "\n";
    std::cout << "  avg same-context corr:  " << summary.avg_same_context_corr << "\n";
    std::cout << "  remapping detected:     " << (summary.remapping_detected ? "YES" : "NO") << "\n";
    std::cout << "  restoration detected:   " << (summary.restoration_detected ? "YES" : "NO") << "\n";

    std::cout << "\n  Computing decoders...\n";
    auto ctx_dec = decode_context(data.activations, data.meta.num_contexts);
    std::cout << "  context decode accuracy:  " << std::fixed << std::setprecision(1)
              << (ctx_dec.test_accuracy * 100) << "% (train: "
              << (ctx_dec.train_accuracy * 100) << "%)\n";

    auto pos_dec = decode_position(data.activations, gw, gh);
    std::cout << "  position decode accuracy: " << std::fixed << std::setprecision(1)
              << (pos_dec.test_accuracy * 100) << "% (train: "
              << (pos_dec.train_accuracy * 100) << "%)\n";

    // ── Generate report ──
    std::cout << "\n  Generating HTML report...\n";
    std::string report_path = generate_report(data);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== Analysis complete ===\n";
    std::cout << "  elapsed: " << std::fixed << std::setprecision(1) << elapsed << " s\n";
    std::cout << "  report: " << report_path << "\n";
    std::cout << "\nOpen the report in a browser to view visualizations.\n";

    return 0;
}
