#include "Decoder.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "../utils/Random.h"

namespace remap2d {

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using IVector = Eigen::VectorXi;

// ── Softmax (row-wise) ──────────────────────────────────────────────────────

static Matrix softmax(const Matrix& logits) {
    Matrix probs = logits;
    for (int i = 0; i < logits.rows(); ++i) {
        float mx = logits.row(i).maxCoeff();
        Vector row = (logits.row(i).transpose().array() - mx).exp();
        probs.row(i) = row.transpose() / row.sum();
    }
    return probs;
}

// ── One-hot encoding ────────────────────────────────────────────────────────

static Matrix one_hot(const IVector& y, int num_classes) {
    Matrix oh = Matrix::Zero(y.size(), num_classes);
    for (int i = 0; i < y.size(); ++i) {
        int label = y(i);
        if (label >= 0 && label < num_classes)
            oh(i, label) = 1.0f;
    }
    return oh;
}

// ── Predict ─────────────────────────────────────────────────────────────────

static IVector predict(const Matrix& X, const Matrix& W, const Vector& b) {
    Matrix logits = X * W;
    for (int i = 0; i < X.rows(); ++i)
        logits.row(i) += b.transpose();
    IVector preds(X.rows());
    for (int i = 0; i < X.rows(); ++i)
        logits.row(i).maxCoeff(&preds(i));
    return preds;
}

// ── Accuracy ────────────────────────────────────────────────────────────────

static float accuracy(const IVector& preds, const IVector& labels) {
    if (preds.size() == 0) return 0.0f;
    int correct = 0;
    for (int i = 0; i < preds.size(); ++i)
        if (preds(i) == labels(i)) ++correct;
    return static_cast<float>(correct) / preds.size();
}

// ── Main decoder training ───────────────────────────────────────────────────

DecoderResult train_softmax_decoder(
    const std::vector<std::vector<float>>& X_in,
    const std::vector<int>& y_in,
    int num_classes,
    int max_iter,
    float lr,
    float l2,
    unsigned int seed)
{
    DecoderResult result{0, 0, 0, 0, 0, 0, 0};

    int N = static_cast<int>(X_in.size());
    if (N == 0 || y_in.size() != X_in.size()) return result;
    int D = static_cast<int>(X_in[0].size());

    result.num_samples  = N;
    result.num_classes  = num_classes;
    result.num_features = D;

    // Build Eigen matrices
    Matrix X(N, D);
    IVector y(N);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d)
            X(i, d) = X_in[i][d];
        y(i) = y_in[i];
    }

    // Shuffle indices for train/test split
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    Random rng(seed);
    rng.shuffle(idx);

    int n_train = static_cast<int>(N * 0.8f);
    if (n_train < 1) n_train = 1;
    if (n_train >= N) n_train = N - 1;
    int n_test = N - n_train;
    result.num_train = n_train;
    result.num_test  = n_test;

    Matrix X_tr(n_train, D), X_te(n_test, D);
    IVector y_tr(n_train), y_te(n_test);
    for (int i = 0; i < n_train; ++i) {
        X_tr.row(i) = X.row(idx[i]);
        y_tr(i)     = y(idx[i]);
    }
    for (int i = 0; i < n_test; ++i) {
        X_te.row(i) = X.row(idx[n_train + i]);
        y_te(i)     = y(idx[n_train + i]);
    }

    // Initialize weights
    float init_std = std::sqrt(1.0f / D);
    Matrix W(D, num_classes);
    Vector b = Vector::Zero(num_classes);
    for (int j = 0; j < num_classes; ++j)
        for (int d = 0; d < D; ++d)
            W(d, j) = rng.normal() * init_std;

    // One-hot targets
    Matrix Y_tr = one_hot(y_tr, num_classes);

    // Gradient descent
    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix logits = X_tr * W;
        for (int i = 0; i < n_train; ++i) logits.row(i) += b.transpose();
        Matrix probs = softmax(logits);

        Matrix diff = probs - Y_tr;  // [n_train, num_classes]
        Matrix dW = X_tr.transpose() * diff / static_cast<float>(n_train);
        dW += l2 * W;
        Vector db = diff.colwise().sum().transpose() / static_cast<float>(n_train);

        W -= lr * dW;
        b -= lr * db;
    }

    // Evaluate
    IVector pred_tr = predict(X_tr, W, b);
    IVector pred_te = predict(X_te, W, b);
    result.train_accuracy = accuracy(pred_tr, y_tr);
    result.test_accuracy  = accuracy(pred_te, y_te);

    return result;
}

// ── Convenience wrappers ────────────────────────────────────────────────────

DecoderResult decode_context(const std::vector<LoadedActivation>& activations,
                             int num_contexts)
{
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    X.reserve(activations.size());
    y.reserve(activations.size());
    for (const auto& a : activations) {
        X.push_back(a.hidden);
        y.push_back(a.context_id);
    }
    return train_softmax_decoder(X, y, num_contexts);
}

DecoderResult decode_position(const std::vector<LoadedActivation>& activations,
                              int grid_w, int grid_h)
{
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    X.reserve(activations.size());
    y.reserve(activations.size());
    int num_cells = grid_w * grid_h;
    for (const auto& a : activations) {
        X.push_back(a.hidden);
        y.push_back(a.pos_y * grid_w + a.pos_x);
    }
    return train_softmax_decoder(X, y, num_cells);
}

DecoderResult decode_position_in_context(
    const std::vector<LoadedActivation>& activations,
    int context_id, int grid_w, int grid_h)
{
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    int num_cells = grid_w * grid_h;
    for (const auto& a : activations) {
        if (a.context_id != context_id) continue;
        X.push_back(a.hidden);
        y.push_back(a.pos_y * grid_w + a.pos_x);
    }
    return train_softmax_decoder(X, y, num_cells);
}

} // namespace remap2d
