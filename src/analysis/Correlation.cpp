#include "Correlation.h"
#include <cmath>
#include <algorithm>
#include <set>

namespace remap2d {

// ── Block extraction ────────────────────────────────────────────────────────

std::vector<Block> extract_blocks(const std::vector<EpisodeMeta>& episodes) {
    std::vector<Block> blocks;
    if (episodes.empty()) return blocks;

    int block_id = 0;
    int start = episodes[0].episode_id;
    int ctx = episodes[0].context_id;

    for (size_t i = 1; i < episodes.size(); ++i) {
        if (episodes[i].context_id != ctx) {
            blocks.push_back({block_id++, ctx, start, (int)episodes[i-1].episode_id});
            start = episodes[i].episode_id;
            ctx = episodes[i].context_id;
        }
    }
    // Last block
    blocks.push_back({block_id, ctx, start, (int)episodes.back().episode_id});
    return blocks;
}

// ── Rate maps for a block ───────────────────────────────────────────────────

ContextRateMaps compute_block_rate_maps(
    const std::vector<LoadedActivation>& activations,
    const Block& block, int hidden_size, int grid_w, int grid_h)
{
    ContextRateMaps m;
    m.context_id  = block.context_id;
    m.hidden_size = hidden_size;
    m.grid_w      = grid_w;
    m.grid_h      = grid_h;

    int cells = grid_w * grid_h;
    m.maps.assign(hidden_size * cells, 0.0f);
    m.visits.assign(cells, 0);

    for (const auto& a : activations) {
        if (a.episode_id < block.start_episode || a.episode_id > block.end_episode)
            continue;
        int cell = a.pos_y * grid_w + a.pos_x;
        if (cell < 0 || cell >= cells) continue;
        m.visits[cell]++;
        for (int u = 0; u < hidden_size; ++u) {
            m.maps[u * cells + cell] += a.hidden[u];
        }
    }

    // Average
    for (int cell = 0; cell < cells; ++cell) {
        if (m.visits[cell] > 0) {
            float inv = 1.0f / static_cast<float>(m.visits[cell]);
            for (int u = 0; u < hidden_size; ++u) {
                m.maps[u * cells + cell] *= inv;
            }
        }
    }
    return m;
}

// ── Pearson correlation ─────────────────────────────────────────────────────

float pearson(const std::vector<float>& a, const std::vector<float>& b) {
    int n = static_cast<int>(a.size());
    if (n != (int)b.size() || n == 0) return 0.0f;

    float mean_a = 0.0f, mean_b = 0.0f;
    for (int i = 0; i < n; ++i) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= n; mean_b /= n;

    float cov = 0.0f, var_a = 0.0f, var_b = 0.0f;
    for (int i = 0; i < n; ++i) {
        float da = a[i] - mean_a;
        float db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    float denom = std::sqrt(var_a * var_b);
    if (denom < 1e-12f) return 0.0f;
    return cov / denom;
}

// ── Population vector correlation ───────────────────────────────────────────

float population_vector_correlation(const ContextRateMaps& map_a,
                                     const ContextRateMaps& map_b)
{
    int hs = std::min(map_a.hidden_size, map_b.hidden_size);
    int cells = map_a.grid_w * map_a.grid_h;
    if (hs == 0 || cells == 0) return 0.0f;

    float corr_sum = 0.0f;
    int shared_count = 0;

    for (int c = 0; c < cells; ++c) {
        if (map_a.visits[c] == 0 || map_b.visits[c] == 0) continue;

        std::vector<float> pv_a(hs), pv_b(hs);
        for (int u = 0; u < hs; ++u) {
            pv_a[u] = map_a.maps[u * cells + c];
            pv_b[u] = map_b.maps[u * cells + c];
        }
        float corr = pearson(pv_a, pv_b);
        if (!std::isnan(corr)) {
            corr_sum += corr;
            shared_count++;
        }
    }

    if (shared_count == 0) return 0.0f;
    return corr_sum / shared_count;
}

// ── Full correlation matrix ─────────────────────────────────────────────────

std::vector<std::vector<float>> compute_correlation_matrix(
    const std::vector<ContextRateMaps>& block_maps)
{
    int n = static_cast<int>(block_maps.size());
    std::vector<std::vector<float>> mat(n, std::vector<float>(n, 0.0f));

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float corr = (i == j) ? 1.0f :
                population_vector_correlation(block_maps[i], block_maps[j]);
            mat[i][j] = corr;
            mat[j][i] = corr;
        }
    }
    return mat;
}

// ── Remapping summary ───────────────────────────────────────────────────────

RemappingSummary analyze_remapping(
    const std::vector<Block>& blocks,
    const std::vector<std::vector<float>>& corr_matrix)
{
    RemappingSummary s;
    s.avg_cross_context_corr = 0.0f;
    s.avg_same_context_corr  = 0.0f;
    s.remapping_detected     = false;
    s.restoration_detected   = false;
    s.num_blocks             = static_cast<int>(blocks.size());

    std::set<int> ctx_set;
    for (const auto& b : blocks) ctx_set.insert(b.context_id);
    s.num_contexts = static_cast<int>(ctx_set.size());

    float cross_sum = 0.0f, same_sum = 0.0f;
    int cross_count = 0, same_count = 0;

    for (int i = 0; i < (int)blocks.size(); ++i) {
        for (int j = i + 1; j < (int)blocks.size(); ++j) {
            if (blocks[i].context_id == blocks[j].context_id) {
                same_sum += corr_matrix[i][j];
                same_count++;
            } else {
                cross_sum += corr_matrix[i][j];
                cross_count++;
            }
        }
    }

    if (cross_count > 0) s.avg_cross_context_corr = cross_sum / cross_count;
    if (same_count > 0)  s.avg_same_context_corr  = same_sum  / same_count;

    s.remapping_detected   = (s.avg_cross_context_corr < 0.3f);
    s.restoration_detected = (s.avg_same_context_corr > 0.5f);

    return s;
}

} // namespace remap2d
