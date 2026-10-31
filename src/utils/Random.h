#ifndef REMAP2D_RANDOM_H
#define REMAP2D_RANDOM_H

#include <random>
#include <cstdint>

namespace remap2d {

// Seeded RNG wrapper for reproducibility.
// All randomness in the project goes through this class.
class Random {
public:
    explicit Random(unsigned int seed) : rng_(seed) {}

    void seed(unsigned int s) { rng_.seed(s); }

    // Uniform float in [0, 1)
    float uniform() { return dist_(rng_); }

    // Uniform int in [lo, hi] inclusive
    int randint(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng_);
    }

    // Standard normal float (mean 0, std 1)
    float normal() { return norm_(rng_); }

    // True with probability p
    bool bernoulli(float p) { return uniform() < p; }

    // Shuffle a vector in-place (Fisher-Yates)
    template<typename T>
    void shuffle(std::vector<T>& v) {
        for (int i = static_cast<int>(v.size()) - 1; i > 0; --i) {
            int j = randint(0, i);
            std::swap(v[i], v[j]);
        }
    }

    // Sample an index from a probability distribution (normalised)
    int sample_categorical(const std::vector<float>& probs) {
        float u = uniform();
        float cum = 0.0f;
        for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
            cum += probs[i];
            if (u < cum) return i;
        }
        return static_cast<int>(probs.size()) - 1;
    }

    std::mt19937& engine() { return rng_; }

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_{0.0f, 1.0f};
    std::normal_distribution<float>       norm_{0.0f, 1.0f};
};

} // namespace remap2d

#endif // REMAP2D_RANDOM_H
