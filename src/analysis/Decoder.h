#ifndef REMAP2D_DECODER_H
#define REMAP2D_DECODER_H

#include <vector>
#include "RunLoader.h"

namespace remap2d {

// Result of training a softmax decoder.
struct DecoderResult {
    float train_accuracy;
    float test_accuracy;
    int   num_samples;
    int   num_classes;
    int   num_features;
    int   num_train;
    int   num_test;
};

// Train a multinomial logistic regression (softmax) decoder.
// Uses gradient descent with L2 regularization.
// 80/20 train/test split with fixed-seed shuffling for reproducibility.
DecoderResult train_softmax_decoder(
    const std::vector<std::vector<float>>& X,   // [N][hidden_size]
    const std::vector<int>& y,                   // [N] class labels
    int num_classes,
    int max_iter = 800,
    float lr = 0.3f,
    float l2 = 1e-4f,
    unsigned int seed = 42);

// Decode context from hidden activations.
DecoderResult decode_context(const std::vector<LoadedActivation>& activations,
                             int num_contexts);

// Decode position (cell index) from hidden activations.
DecoderResult decode_position(const std::vector<LoadedActivation>& activations,
                              int grid_w, int grid_h);

// Decode position for a single context only (within-context position code).
DecoderResult decode_position_in_context(
    const std::vector<LoadedActivation>& activations,
    int context_id, int grid_w, int grid_h);

} // namespace remap2d

#endif // REMAP2D_DECODER_H
