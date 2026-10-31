#ifndef REMAP2D_CORRELATION_H
#define REMAP2D_CORRELATION_H

#include <vector>
#include "RunLoader.h"
#include "RateMaps.h"

namespace remap2d {

// A "block" is a contiguous run of episodes in the same context.
// In an ABA schedule there are 3 blocks: A1, B, A2.
struct Block {
    int block_id;
    int context_id;
    int start_episode;  // inclusive
    int end_episode;    // inclusive
};

// Extract blocks from the episode list (consecutive same-context episodes).
std::vector<Block> extract_blocks(const std::vector<EpisodeMeta>& episodes);

// Compute rate maps for a specific block (subset of activations).
ContextRateMaps compute_block_rate_maps(
    const std::vector<LoadedActivation>& activations,
    const Block& block, int hidden_size, int grid_w, int grid_h);

// Pearson correlation between two vectors.
float pearson(const std::vector<float>& a, const std::vector<float>& b);

// Population vector correlation between two rate map sets.
// For each cell visited in both maps, compute the correlation between
// the population vectors (all units' rates at that cell), then average.
// Low correlation → global remapping.
float population_vector_correlation(const ContextRateMaps& map_a,
                                     const ContextRateMaps& map_b);

// Full correlation matrix between all pairs of blocks.
// result[i][j] = correlation between block i and block j.
std::vector<std::vector<float>> compute_correlation_matrix(
    const std::vector<ContextRateMaps>& block_maps);

// Summary of remapping findings.
struct RemappingSummary {
    // Average cross-context correlation (between blocks with different context_id)
    float avg_cross_context_corr;
    // Average same-context correlation (between blocks with same context_id, e.g. A1 vs A2)
    float avg_same_context_corr;
    // Did remapping occur? (cross-context corr < 0.3)
    bool  remapping_detected;
    // Was the representation restored? (same-context corr > 0.5)
    bool  restoration_detected;
    // Number of blocks
    int   num_blocks;
    // Number of distinct contexts
    int   num_contexts;
};

RemappingSummary analyze_remapping(
    const std::vector<Block>& blocks,
    const std::vector<std::vector<float>>& corr_matrix);

} // namespace remap2d

#endif // REMAP2D_CORRELATION_H
