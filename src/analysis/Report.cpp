#include "Report.h"
#include "RateMaps.h"
#include "Correlation.h"
#include "Decoder.h"
#include "../utils/Svg.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace remap2d {

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::string fmt(float v, int prec = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

// ── Metadata table ──────────────────────────────────────────────────────────

static std::string metadata_table(const RunData& data) {
    std::ostringstream h;
    h << "<table class=\"meta\">\n";
    h << "<tr><th>Field</th><th>Value</th></tr>\n";
    h << "<tr><td>Run ID</td><td>" << xml_escape(data.meta.run_id) << "</td></tr>\n";
    h << "<tr><td>Timestamp</td><td>" << xml_escape(data.meta.timestamp) << "</td></tr>\n";
    h << "<tr><td>Seed</td><td>" << data.meta.seed << "</td></tr>\n";
    h << "<tr><td>Replay Enabled</td><td>" << (data.meta.replay_enabled ? "Yes" : "No") << "</td></tr>\n";
    h << "<tr><td>Hidden Size</td><td>" << data.meta.hidden_size << "</td></tr>\n";
    h << "<tr><td>Num Contexts</td><td>" << data.meta.num_contexts << "</td></tr>\n";
    h << "<tr><td>Episodes</td><td>" << data.episodes.size() << "</td></tr>\n";
    h << "<tr><td>Activation Samples</td><td>" << data.activations.size() << "</td></tr>\n";
    h << "<tr><td>Config Hash</td><td><code>" << xml_escape(data.meta.config_hash) << "</code></td></tr>\n";
    h << "</table>\n";

    // Context list
    h << "<h3>Contexts</h3>\n<table class=\"meta\">\n";
    h << "<tr><th>Context ID</th><th>Maze Name</th><th>Grid</th></tr>\n";
    for (const auto& c : data.contexts) {
        h << "<tr><td>" << c.context_id << "</td><td>" << xml_escape(c.maze_name)
          << "</td><td>" << c.grid_w << " x " << c.grid_h << "</td></tr>\n";
    }
    h << "</table>\n";
    return h.str();
}

// ── Learning curves ─────────────────────────────────────────────────────────

static std::string learning_curves(const RunData& data,
                                    const std::vector<Block>& blocks)
{
    if (data.episodes.empty()) return "<p>No episode data.</p>";

    // Group rewards by block
    std::vector<std::vector<float>> series;
    std::vector<std::string> names, colors;
    static const char* palette[] = {"#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"};

    for (size_t b = 0; b < blocks.size(); ++b) {
        std::vector<float> rewards;
        for (const auto& ep : data.episodes) {
            if (ep.episode_id >= blocks[b].start_episode &&
                ep.episode_id <= blocks[b].end_episode) {
                rewards.push_back(ep.total_reward);
            }
        }
        if (!rewards.empty()) {
            // Moving average (window 10)
            std::vector<float> smoothed;
            int win = 10;
            for (int i = 0; i < (int)rewards.size(); ++i) {
                int lo = std::max(0, i - win / 2);
                int hi = std::min((int)rewards.size() - 1, i + win / 2);
                float sum = 0.0f;
                for (int j = lo; j <= hi; ++j) sum += rewards[j];
                smoothed.push_back(sum / (hi - lo + 1));
            }
            series.push_back(smoothed);
            names.push_back("Block " + std::to_string(b) + " (ctx " +
                            std::to_string(blocks[b].context_id) + ")");
            colors.push_back(palette[b % 5]);
        }
    }

    // Find y range
    float y_min = 1e9, y_max = -1e9;
    for (const auto& s : series)
        for (float v : s) { y_min = std::min(y_min, v); y_max = std::max(y_max, v); }
    if (y_min > y_max) { y_min = -1; y_max = 1; }
    // Pad
    float pad = (y_max - y_min) * 0.1f;
    y_min -= pad; y_max += pad;

    // Find x range
    int max_len = 0;
    for (const auto& s : series) max_len = std::max(max_len, (int)s.size());

    std::string svg = svg_line_chart(series, names, colors,
                                      0, static_cast<float>(max_len),
                                      y_min, y_max,
                                      700, 300, "Episode (within block)", "Reward");
    return svg;
}

// ── Rate maps section ───────────────────────────────────────────────────────

static std::string rate_maps_section(const std::vector<ContextRateMaps>& block_maps,
                                      const std::vector<Block>& blocks)
{
    if (block_maps.empty()) return "<p>No activation data for rate maps.</p>";

    std::ostringstream h;

    // Find top tuned units from the first block
    int top_n = 8;
    std::vector<int> top_units = find_most_tuned_units(block_maps[0], top_n);

    // Compute global min/max for consistent colormap across all maps
    float global_min = 1e9, global_max = -1e9;
    int cells = block_maps[0].grid_w * block_maps[0].grid_h;
    for (const auto& bm : block_maps) {
        for (int u : top_units) {
            for (int c = 0; c < cells; ++c) {
                if (bm.visits[c] > 0) {
                    float v = bm.maps[u * cells + c];
                    global_min = std::min(global_min, v);
                    global_max = std::max(global_max, v);
                }
            }
        }
    }
    if (global_min > global_max) { global_min = 0; global_max = 1; }

    // Build a table: rows = blocks, columns = top units
    int cell_size = 24;
    h << "<p>Showing the top " << top_n << " most spatially tuned hidden units "
      << "(by Skaggs spatial information score) from block 0. "
      << "Each cell is a 10x10 spatial rate map; warmer colors = higher activation. "
      << "Compare columns to see how the same unit's representation changes across blocks.</p>\n";

    h << "<div class=\"rate-map-grid\">\n";
    h << "<table class=\"rate-maps\">\n";

    // Header row with unit numbers
    h << "<tr><th>Block</th>";
    for (int u : top_units)
        h << "<th>Unit " << u << "</th>";
    h << "</tr>\n";

    for (size_t b = 0; b < block_maps.size(); ++b) {
        const auto& bm = block_maps[b];
        h << "<tr><td class=\"block-label\">Block " << b << "<br>ctx " << blocks[b].context_id
          << "</td>";
        for (int u : top_units) {
            // Extract this unit's rate map
            std::vector<float> unit_map(bm.grid_h * bm.grid_w, 0.0f);
            for (int c = 0; c < cells; ++c)
                unit_map[c] = bm.maps[u * cells + c];
            // Mark unvisited cells as min
            for (int c = 0; c < cells; ++c)
                if (bm.visits[c] == 0) unit_map[c] = global_min - 0.01f;

            std::string svg = svg_heatmap(unit_map, bm.grid_h, bm.grid_w,
                                           global_min, global_max, cell_size, true);
            h << "<td>" << svg << "</td>";
        }
        h << "</tr>\n";
    }
    h << "</table>\n</div>\n";

    return h.str();
}

// ── Correlation section ─────────────────────────────────────────────────────

static std::string correlation_section(const std::vector<Block>& blocks,
                                        const std::vector<std::vector<float>>& corr_matrix,
                                        const RemappingSummary& summary)
{
    std::ostringstream h;

    // Labels for matrix
    std::vector<std::string> labels;
    for (size_t i = 0; i < blocks.size(); ++i)
        labels.push_back("B" + std::to_string(i) + "(ctx" + std::to_string(blocks[i].context_id) + ")");

    h << "<div class='corr-container'>\n";
    h << svg_correlation_matrix(labels, corr_matrix, 70);
    h << "</div>\n";

    h << "<h3>Interpretation</h3>\n<ul>\n";
    h << "<li><strong>Avg cross-context correlation:</strong> " << fmt(summary.avg_cross_context_corr)
      << " (low values indicate global remapping)</li>\n";
    h << "<li><strong>Avg same-context correlation:</strong> " << fmt(summary.avg_same_context_corr)
      << " (high values indicate representation restoration, e.g. A1 vs A2)</li>\n";
    h << "<li><strong>Remapping detected:</strong> "
      << (summary.remapping_detected ? "<span class='yes'>Yes</span>" : "<span class='no'>No</span>")
      << " (threshold: cross-context corr < 0.3)</li>\n";
    h << "<li><strong>Restoration detected:</strong> "
      << (summary.restoration_detected ? "<span class='yes'>Yes</span>" : "<span class='no'>No</span>")
      << " (threshold: same-context corr > 0.5)</li>\n";
    h << "</ul>\n";

    return h.str();
}

// ── Decoder section ─────────────────────────────────────────────────────────

static std::string decoder_section(const RunData& data) {
    std::ostringstream h;

    // Context decoding
    DecoderResult ctx_result = decode_context(data.activations, data.meta.num_contexts);

    // Position decoding (all contexts)
    DecoderResult pos_result = decode_position(data.activations, 10, 10);

    // Per-context position decoding
    std::vector<std::string> pos_labels;
    std::vector<float> pos_values;
    for (const auto& c : data.contexts) {
        DecoderResult r = decode_position_in_context(data.activations, c.context_id, c.grid_w, c.grid_h);
        pos_labels.push_back("Ctx " + std::to_string(c.context_id));
        pos_values.push_back(r.test_accuracy);
    }

    h << "<h3>Context Decoding</h3>\n";
    h << "<p>Can a linear decoder predict which context the agent is in from hidden activations?</p>\n";
    h << "<table class=\"meta\">\n";
    h << "<tr><th>Metric</th><th>Value</th></tr>\n";
    h << "<tr><td>Train accuracy</td><td>" << fmt(ctx_result.train_accuracy * 100, 1) << "%</td></tr>\n";
    h << "<tr><td>Test accuracy</td><td>" << fmt(ctx_result.test_accuracy * 100, 1) << "%</td></tr>\n";
    h << "<tr><td>Num samples</td><td>" << ctx_result.num_samples << "</td></tr>\n";
    h << "<tr><td>Num classes</td><td>" << ctx_result.num_classes << "</td></tr>\n";
    h << "</table>\n";

    h << "<h3>Position Decoding (per context)</h3>\n";
    h << "<p>Can a linear decoder predict the agent's position from hidden activations? "
      << "Decoded separately within each context.</p>\n";

    float max_val = 1.0f;
    for (float v : pos_values) max_val = std::max(max_val, v);
    h << svg_bar_chart(pos_labels, pos_values, max_val, 500, 250, "#4a90d9");

    h << "<h3>Overall Position Decoding (across all contexts)</h3>\n";
    h << "<table class=\"meta\">\n";
    h << "<tr><th>Metric</th><th>Value</th></tr>\n";
    h << "<tr><td>Train accuracy</td><td>" << fmt(pos_result.train_accuracy * 100, 1) << "%</td></tr>\n";
    h << "<tr><td>Test accuracy</td><td>" << fmt(pos_result.test_accuracy * 100, 1) << "%</td></tr>\n";
    h << "<tr><td>Num classes</td><td>" << pos_result.num_classes << " (grid cells)</td></tr>\n";
    h << "</table>\n";

    h << "<h3>Interpretation</h3>\n<ul>\n";
    if (ctx_result.test_accuracy > 0.7f && pos_result.test_accuracy > 0.3f) {
        h << "<li><strong>Integrated code:</strong> Both context and position are decodable "
          << "from the same hidden units, suggesting an integrated representation.</li>\n";
    } else if (pos_result.test_accuracy > 0.3f && ctx_result.test_accuracy < 0.5f) {
        h << "<li><strong>Segmented code:</strong> Position is decodable but context is not, "
          << "suggesting a position-only code without context integration.</li>\n";
    } else {
        h << "<li><strong>Mixed code:</strong> Partial decodability of both context and position.</li>\n";
    }
    h << "</ul>\n";

    return h.str();
}

// ── Summary text ────────────────────────────────────────────────────────────

static std::string summary_text(const RunData& data,
                                 const RemappingSummary& remap,
                                 const DecoderResult& ctx_dec,
                                 const DecoderResult& pos_dec)
{
    std::ostringstream h;
    h << "<div class=\"summary-box\">\n";
    h << "<h3>Key Findings</h3>\n<ol>\n";

    // 1. Remapping
    h << "<li><strong>Global remapping: </strong>";
    if (remap.remapping_detected) {
        h << "Yes — the population vector correlation between different contexts is "
          << fmt(remap.avg_cross_context_corr) << ", indicating that the hidden-layer "
          << "representations reorganize substantially when the agent moves between "
          << "maze contexts, analogous to hippocampal global remapping.";
    } else {
        h << "Inconclusive — the cross-context correlation is "
          << fmt(remap.avg_cross_context_corr) << ", which does not meet the remapping "
          << "threshold (< 0.3). The representations may be too similar across contexts, "
          << "or the agent has not developed sufficiently distinct spatial representations.";
    }
    h << "</li>\n";

    // 2. Restoration
    h << "<li><strong>Representation restoration (A1 vs A2): </strong>";
    if (remap.avg_same_context_corr > 0.0f) {
        h << "The same-context correlation is " << fmt(remap.avg_same_context_corr)
          << ". ";
        if (remap.restoration_detected) {
            h << "This indicates that the representation is substantially restored "
              << "when returning to a familiar context.";
        } else {
            h << "The representation is only partially restored when returning to "
              << "a familiar context.";
        }
    } else {
        h << "Not applicable (no repeated context in the schedule).";
    }
    h << "</li>\n";

    // 3. Replay effect
    h << "<li><strong>Replay condition: </strong>";
    if (data.meta.replay_enabled) {
        h << "Experience replay was <strong>enabled</strong>. Compare this report with "
          << "the replay-off condition to assess whether replay improves flexibility.";
    } else {
        h << "Experience replay was <strong>disabled</strong>. Compare with the replay-on "
          << "condition to see if replay improves remapping or restoration.";
    }
    h << "</li>\n";

    // 4. Decoding
    h << "<li><strong>Decoding: </strong>Context accuracy = "
      << fmt(ctx_dec.test_accuracy * 100, 1) << "%, position accuracy = "
      << fmt(pos_dec.test_accuracy * 100, 1) << "%.</li>\n";

    h << "</ol>\n";

    // Honest scoping
    h << "<h3>Scope and Limitations</h3>\n<ul>\n";
    h << "<li>This is a computational analogy study, not a claim about real hippocampal tissue.</li>\n";
    h << "<li>The core remapping effect has been demonstrated in prior 3D deep-RL work; "
      << "this project tests whether it generalizes to a 2D grid-maze setting.</li>\n";
    h << "<li>Results are sensitive to hyperparameters and the number of contexts. "
      << "Conclusions should be drawn from multiple seeds and configurations.</li>\n";
    h << "<li>Small network (MLP) and limited maze variety may constrain the generality "
      << "of findings.</li>\n";
    h << "</ul>\n";
    h << "</div>\n";

    return h.str();
}

// ── Main report generation ──────────────────────────────────────────────────

std::string generate_report(const RunData& data) {
    // Compute analysis
    auto blocks = extract_blocks(data.episodes);

    int hs = data.meta.hidden_size;
    int gw = data.contexts.empty() ? 10 : data.contexts[0].grid_w;
    int gh = data.contexts.empty() ? 10 : data.contexts[0].grid_h;

    std::vector<ContextRateMaps> block_maps;
    for (const auto& b : blocks) {
        block_maps.push_back(compute_block_rate_maps(data.activations, b, hs, gw, gh));
    }

    auto corr_matrix = compute_correlation_matrix(block_maps);
    auto remap_summary = analyze_remapping(blocks, corr_matrix);

    DecoderResult ctx_dec = decode_context(data.activations, data.meta.num_contexts);
    DecoderResult pos_dec = decode_position(data.activations, gw, gh);

    // Assemble HTML
    std::ostringstream html;
    html << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n";
    html << "<meta charset=\"UTF-8\">\n";
    html << "<title>Remapping Report: " << xml_escape(data.meta.run_id) << "</title>\n";
    html << "<style>\n";
    html << "  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; "
          << "background: #fafafa; color: #222; max-width: 1200px; }\n";
    html << "  h1 { color: #1a3a5c; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }\n";
    html << "  h2 { color: #1a3a5c; margin-top: 30px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }\n";
    html << "  h3 { color: #333; margin-top: 20px; }\n";
    html << "  .section { background: white; padding: 20px; margin: 15px 0; "
          << "border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }\n";
    html << "  table.meta { border-collapse: collapse; margin: 10px 0; }\n";
    html << "  table.meta th, table.meta td { border: 1px solid #ddd; padding: 6px 12px; "
          << "text-align: left; }\n";
    html << "  table.meta th { background: #e8eef5; }\n";
    html << "  table.rate-maps { border-collapse: collapse; }\n";
    html << "  table.rate-maps th { font-size: 11px; padding: 4px; text-align: center; }\n";
    html << "  table.rate-maps td { padding: 4px; text-align: center; }\n";
    html << "  td.block-label { font-size: 12px; font-weight: bold; text-align: right; padding-right: 8px; }\n";
    html << "  .summary-box { background: #f0f7ff; padding: 15px; border-left: 4px solid #4a90d9; }\n";
    html << "  .yes { color: #2d7a2d; font-weight: bold; }\n";
    html << "  .no { color: #c0392b; font-weight: bold; }\n";
    html << "  code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 13px; }\n";
    html << "  svg { display: block; margin: 10px 0; }\n";
    html << "</style>\n";
    html << "</head>\n<body>\n";

    html << "<h1>Global Remapping in a 2D RL Agent</h1>\n";
    html << "<p><em>Replication-and-simplification study: does a lightweight 2D grid-maze "
          << "DQN agent exhibit hippocampal-like global remapping, and does experience "
          << "replay improve flexibility?</em></p>\n";

    html << "<div class=\"section\">\n";
    html << "<h2>1. Run Metadata</h2>\n";
    html << metadata_table(data);
    html << "</div>\n";

    html << "<div class=\"section\">\n";
    html << "<h2>2. Learning Curves</h2>\n";
    html << "<p>Smoothed (window=10) total reward per episode, split by context block. "
          << "Each line shows learning within one block of the ABA schedule.</p>\n";
    html << learning_curves(data, blocks);
    html << "</div>\n";

    html << "<div class=\"section\">\n";
    html << "<h2>3. Rate Maps (Spatial Tuning of Hidden Units)</h2>\n";
    html << rate_maps_section(block_maps, blocks);
    html << "</div>\n";

    html << "<div class=\"section\">\n";
    html << "<h2>4. Cross-Context Population Vector Correlation</h2>\n";
    html << "<p>Correlation between population vectors (all units' rates at each visited cell) "
          << "across blocks. Low off-diagonal values indicate global remapping; "
          << "high same-context values (e.g., B0 vs B2 if both are context A) indicate "
          << "representation restoration.</p>\n";
    html << correlation_section(blocks, corr_matrix, remap_summary);
    html << "</div>\n";

    html << "<div class=\"section\">\n";
    html << "<h2>5. Linear Decoding of Context and Position</h2>\n";
    html << decoder_section(data);
    html << "</div>\n";

    html << "<div class=\"section\">\n";
    html << "<h2>6. Summary</h2>\n";
    html << summary_text(data, remap_summary, ctx_dec, pos_dec);
    html << "</div>\n";

    html << "<hr>\n";
    html << "<p style=\"font-size: 11px; color: #888;\">Generated by remap2d analysis tool. "
          << "All analysis in C++ — no Python dependencies.</p>\n";
    html << "</body>\n</html>\n";

    // Write to file
    std::string report_path = data.run_dir + "/report.html";
    std::ofstream ofs(report_path);
    if (!ofs) throw std::runtime_error("Cannot write report to " + report_path);
    ofs << html.str();
    ofs.close();

    return report_path;
}

} // namespace remap2d
