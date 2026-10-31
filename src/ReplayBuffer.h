#ifndef REMAP2D_REPLAY_BUFFER_H
#define REMAP2D_REPLAY_BUFFER_H

#include <vector>
#include "DQNAgent.h"
#include "utils/Random.h"

namespace remap2d {

// Ring-buffer replay buffer with an on/off toggle.
// When disabled, add() is a no-op and size() stays 0 — the no-replay ablation.
class ReplayBuffer {
public:
    ReplayBuffer(int capacity, bool enabled, Random& rng)
        : capacity_(capacity), enabled_(enabled), rng_(rng), head_(0), size_(0)
    {
        buffer_.reserve(capacity_);
    }

    void add(const Transition& t) {
        if (!enabled_) return;  // truly inert when disabled
        if (static_cast<int>(buffer_.size()) < capacity_) {
            buffer_.push_back(t);
        } else {
            buffer_[head_] = t;  // ring-buffer overwrite
        }
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }

    // Sample min(batch_size, size()) transitions uniformly at random.
    std::vector<Transition> sample(int batch_size) const {
        std::vector<Transition> out;
        if (size_ == 0) return out;
        int n = std::min(batch_size, size_);
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            int idx = rng_.randint(0, size_ - 1);
            out.push_back(buffer_[idx]);
        }
        return out;
    }

    size_t size()    const { return enabled_ ? size_ : 0; }
    bool   enabled() const { return enabled_; }
    int    capacity() const { return capacity_; }

private:
    int capacity_;
    bool enabled_;
    Random& rng_;
    std::vector<Transition> buffer_;
    int head_;
    int size_;
};

} // namespace remap2d

#endif // REMAP2D_REPLAY_BUFFER_H
