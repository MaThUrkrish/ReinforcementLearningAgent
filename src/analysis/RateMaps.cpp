#include "RateMaps.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace remap2d {

std::vector<ContextRateMaps> compute_rate_maps(
    const std::vector<LoadedActivation>& activations,
    const std::vector<ContextMeta>& contexts)
{
    std::vector<ContextRateMaps> result;

    // Group activations by context
    std::map<int, std::vector<const LoadedActivation*>> by_context;
    for (const auto& a : activations)
        by_context[a.context_id].push_back(&a);

    for (const auto& ctx : contexts) {
        ContextRateMaps crm;
        crm.context_id  = ctx.context_id;
        crm.grid_w      = ctx.grid_w;
        crm.grid_h      = ctx.grid_h;

        // Determine hidden_size from first activation in this context
        int hs = 0;
        auto it = by_context.find(ctx.context_id);
        if (it != by_context.end() && !it->second.empty()) {
            hs = static_cast<int>(it->second[0]->hidden.size());
        }
        crm.hidden_size = hs;

        int cells = crm.grid_w * crm.grid_h;
        crm.maps.assign(hs * cells, 0.0f);
        crm.visits.assign(cells, 0);

        if (it != by_context.end()) {
            for (const auto* a : it->second) {
                int cell = a->pos_y * crm.grid_w + a->pos_x;
                if (cell < 0 || cell >= cells) continue;  // safety
                crm.visits[cell]++;
                for (int u = 0; u < hs; ++u) {
                    crm.maps[u * cells + cell] += a->hidden[u];
                }
            }
        }

        // Average: divide by visit count
        for (int cell = 0; cell < cells; ++cell) {
            if (crm.visits[cell] > 0) {
                float inv = 1.0f / static_cast<float>(crm.visits[cell]);
                for (int u = 0; u < hs; ++u) {
                    crm.maps[u * cells + cell] *= inv;
                }
            }
        }

        result.push_back(std::move(crm));
    }

    return result;
}

// ── Spatial information (Skaggs formula) ────────────────────────────────────
// SI = (1/N) * sum_x lambda(x) * log2(lambda(x) / lambda_bar)
// where lambda(x) = rate at x, lambda_bar = overall mean rate.
// Higher SI = more spatially selective.

float spatial_information(const ContextRateMaps& m, int unit) {
    int cells = m.grid_h * m.grid_w;
    float total_rate = 0.0f;
    int visited = 0;
    for (int c = 0; c < cells; ++c) {
        if (m.visits[c] > 0) {
            total_rate += m.maps[unit * cells + c];
            visited++;
        }
    }
    if (visited == 0) return 0.0f;
    float mean_rate = total_rate / visited;
    if (mean_rate <= 0.0f) return 0.0f;

    float si = 0.0f;
    for (int c = 0; c < cells; ++c) {
        if (m.visits[c] > 0) {
            float r = m.maps[unit * cells + c];
            if (r > 0.0f) {
                si += r * std::log2(r / mean_rate);
            }
        }
    }
    si /= visited;
    return si;
}

std::vector<int> find_most_tuned_units(const ContextRateMaps& maps, int top_n) {
    int hs = maps.hidden_size;
    std::vector<std::pair<float, int>> scores;
    scores.reserve(hs);
    for (int u = 0; u < hs; ++u) {
        scores.push_back({spatial_information(maps, u), u});
    }
    // Sort descending by score
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> result;
    for (int i = 0; i < std::min(top_n, hs); ++i) {
        result.push_back(scores[i].second);
    }
    return result;
}

} // namespace remap2d
