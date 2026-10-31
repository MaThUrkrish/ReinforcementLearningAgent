#ifndef REMAP2D_RATE_MAPS_H
#define REMAP2D_RATE_MAPS_H

#include <vector>
#include <map>
#include "RunLoader.h"

namespace remap2d {

// A rate map for one context: [hidden_size][grid_h][grid_w]
// rate_maps[unit][y][x] = average activation of that unit at (x,y)
struct ContextRateMaps {
    int context_id;
    int hidden_size;
    int grid_w;
    int grid_h;
    // Flattened as [hidden_size * grid_h * grid_w]
    // Index: unit * (grid_h * grid_w) + y * grid_w + x
    std::vector<float> maps;
    // Visit count per cell [grid_h * grid_w]
    std::vector<int> visits;
};

// Compute rate maps for each context from loaded activations.
std::vector<ContextRateMaps> compute_rate_maps(
    const std::vector<LoadedActivation>& activations,
    const std::vector<ContextMeta>& contexts);

// Accessor: rate of unit u at (x,y) in context maps
inline float rate_at(const ContextRateMaps& m, int unit, int x, int y) {
    int idx = unit * (m.grid_h * m.grid_w) + y * m.grid_w + x;
    return m.maps[idx];
}

inline int visits_at(const ContextRateMaps& m, int x, int y) {
    return m.visits[y * m.grid_w + x];
}

// Find the indices of the most spatially tuned units (highest peak-to-mean ratio).
// Returns unit indices sorted by tuning strength, descending.
std::vector<int> find_most_tuned_units(const ContextRateMaps& maps, int top_n);

// Compute the spatial information score (Skaggs) for a single unit.
// Higher = more spatially selective.
float spatial_information(const ContextRateMaps& m, int unit);

} // namespace remap2d

#endif // REMAP2D_RATE_MAPS_H
