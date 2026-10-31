# remap2d — Global Remapping in a Simplified 2D Reinforcement Learning Agent

A replication-and-simplification study: does a lightweight 2D grid-maze DQN agent
exhibit hippocampal-like **global remapping**, and does **experience replay**
improve flexibility?

**100% C++17. No Python. No Python dependencies.**

The original study used a 3D deep-RL agent with visual observations. This project
reduces the problem to a 2D grid-maze with a small MLP, testing whether the same
remapping effect emerges without a rich 3D visual backbone.

---

## Quick Start

### Prerequisites

- **g++** (C++17 support, tested with g++ 14.2)
- **make**
- **libsqlite3-dev** (SQLite development headers)
- Internet access (for fetching Eigen and Catch2 on first build)

### Build

```bash
cd remap2d
make            # builds train_agent, analyze, and test_remap2d
make test       # runs unit tests
```

### Run the Reference Experiment

```bash
# Train with replay ON (ABA schedule, 300 episodes per context)
./train_agent --contexts 0,1,0 --episodes-per-context 300 \
              --hidden-size 64 --replay on --seed 42 --out runs/ref_replay_on

# Train with replay OFF (same config, ablation)
./train_agent --contexts 0,1,0 --episodes-per-context 300 \
              --hidden-size 64 --replay off --seed 42 --out runs/ref_replay_off

# Analyze each run and generate HTML report
./analyze --run runs/ref_replay_on
./analyze --run runs/ref_replay_off

# Open the reports
# runs/ref_replay_on/report.html
# runs/ref_replay_off/report.html
```

Or run everything at once:

```bash
make experiment
```

---

## Architecture

The system is split into two tiers, both in C++:

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│      Training Tier (C++17)       │     │      Analysis Tier (C++17)       │
│                                  │     │                                  │
│  MazeEnvironment  (3 mazes)      │     │  RunLoader     (SQLite reader)   │
│  DQNAgent         (MLP+Adam)     │     │  RateMaps      (spatial tuning)  │
│  ReplayBuffer     (on/off toggle)│     │  Correlation   (pop. vector)     │
│  RunLogger        (SQLite writer)│     │  Decoder       (softmax regr.)   │
│                                  │     │  Report        (HTML + SVG)      │
└──────────────┬───────────────────┘     └──────────────┬───────────────────┘
               │                                        │
               └────────────► runs/*.db ◄──────────────┘
                         (SQLite: metadata + BLOBs)
```

The two tiers communicate only through SQLite files, so analysis can be re-run
without repeating training.

### Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| State encoding | One-hot position + one-hot context | Context cue is analogous to visual cues in the 3D study; without it, the agent cannot distinguish mazes |
| Network | MLP (input → 64 ReLU → 4 Q-values) | Small enough for CPU, large enough for place-cell-like tuning |
| Optimizer | Adam (lr=1e-3) | Robust, minimal tuning |
| Loss | Huber (delta=1.0) | Smooth L1 for stable TD targets |
| Target network | Hard update every 200 steps | Standard DQN stabilization |
| Replay | Ring buffer, capacity 10K, batch 32 | Standard DQN; toggle-able for ablation |
| Storage | SQLite (metadata + BLOBs) | Single inspectable file; no HDF5 dependency |
| Plotting | SVG embedded in HTML | No external plotting library; vector graphics |
| Analysis | C++ with Eigen | Replaces Python/NumPy/scikit-learn entirely |

---

## File Structure

```
remap2d/
├── Makefile                          # Build system (no CMake needed)
├── README.md                         # This file
├── src/
│   ├── main.cpp                      # Training CLI
│   ├── analyze_main.cpp              # Analysis CLI
│   ├── Config.h                      # Configuration + CLI parsing
│   ├── MazeEnvironment.h/.cpp        # 3 maze layouts, grid logic
│   ├── DQNAgent.h/.cpp               # MLP, Adam, Huber, target net
│   ├── ReplayBuffer.h/.cpp           # Ring buffer with on/off toggle
│   ├── RunLogger.h/.cpp              # SQLite persistence
│   ├── utils/
│   │   ├── Random.h/.cpp             # Seeded RNG (mt19937)
│   │   └── Svg.h/.cpp                # SVG generation (heatmap, line, bar)
│   └── analysis/
│       ├── RunLoader.h/.cpp          # Load run from SQLite
│       ├── RateMaps.h/.cpp           # Per-unit spatial rate maps
│       ├── Correlation.h/.cpp        # Population vector correlation
│       ├── Decoder.h/.cpp            # Softmax regression (Eigen)
│       └── Report.h/.cpp             # HTML report assembly
├── tests/
│   └── test_main.cpp                 # Catch2 unit tests (F-01..F-12, I-01..I-05)
├── third_party/
│   ├── eigen_src/                    # Eigen 3.4.0 (fetched on build)
│   └── catch.hpp                     # Catch2 v2.13.10 (single header)
└── runs/                             # Output directory (gitignored)
    ├── ref_replay_on/
    │   ├── run_ref_replay_on.db      # SQLite database
    │   ├── weights.bin               # Saved network weights
    │   ├── config.txt                # Run configuration
    │   └── report.html               # Generated HTML report
    └── ref_replay_off/
        └── ...
```

---

## Usage

### Training CLI

```
./train_agent [OPTIONS]

Options:
  --contexts LIST          Comma-separated context IDs (default: 0,1,0 for ABA)
  --episodes-per-context N Episodes per context block (default: 300)
  --max-steps N            Max steps per episode (default: 200)
  --hidden-size N          Hidden layer size (default: 64)
  --replay on|off          Enable/disable experience replay (default: on)
  --buffer-capacity N      Replay buffer capacity (default: 10000)
  --batch-size N           Minibatch size (default: 32)
  --gamma FLOAT            Discount factor (default: 0.95)
  --lr FLOAT               Learning rate (default: 0.001)
  --target-update-freq N   Target network update frequency in steps (default: 200)
  --log-every N            Log activations every N steps (default: 10)
  --seed N                 Random seed (default: 42)
  --out DIR                Output directory (default: runs/run_001)
  --run-id STRING          Run identifier (default: derived from --out)
```

### Analysis CLI

```
./analyze --run RUN_DIR

Generates:
  RUN_DIR/report.html   (HTML report with embedded SVG visualizations)
```

The report includes:
1. **Run metadata** — configuration, seed, episode/activation counts
2. **Learning curves** — smoothed reward per episode, split by context block
3. **Rate maps** — top 8 most spatially tuned hidden units, per block
4. **Cross-context correlation** — population vector correlation matrix
5. **Linear decoding** — context and position decoder accuracies
6. **Summary** — qualitative findings and limitations

---

## Testing

```bash
make test    # builds and runs test_remap2d
```

Tests cover the functional (F-01 through F-12) and integration (I-01 through I-05)
test cases from the QA plan:

- **F-01..F-03**: Environment reset, wall collision, goal reaching
- **F-04..F-05**: Agent action selection (greedy and random)
- **F-06..F-07**: Replay buffer FIFO eviction and disabled-buffer inertness
- **F-08..F-09**: Logger activation and episode persistence to SQLite
- **F-10**: Rate map recovery from synthetic place field
- **F-11**: Correlation = 1.0 for identical, ≈0 for shuffled
- **F-12**: Decoder ≈100% on linearly separable data
- **I-01**: Full pipeline smoke test (train → log → flush → load → analyze)
- **I-03**: Replay on/off both produce loadable runs
- **I-04**: ABA schedule produces correct context sequence
- **I-05**: Rate map computation is deterministic (idempotent)

---

## Results Summary

Reference experiment: ABA schedule (mazeA → mazeB → mazeA), 300 episodes per
context, hidden_size=64, seed=42.

| Metric | Replay ON | Replay OFF |
|--------|-----------|------------|
| Cross-context correlation | 0.688 | 0.681 |
| Same-context correlation (A1 vs A2) | 0.877 | 0.860 |
| Context decoding accuracy | 100.0% | 100.0% |
| Position decoding accuracy | 33.4% | 31.3% |

**Key findings:**

1. **Partial remapping**: Cross-context correlation (0.68) is lower than
   same-context correlation (0.87), indicating that representations differ
   across mazes but not as dramatically as in the 3D study. The 2D grid-maze
   with shared position encoding limits how much remapping can occur.

2. **Representation restoration**: The high same-context correlation (A1 vs A2
   ≈ 0.87) shows that the representation is substantially restored when
   returning to a familiar context — consistent with hippocampal findings.

3. **Replay effect**: The difference between replay on and off is small in this
   configuration. Replay slightly improves both restoration (0.877 vs 0.860)
   and position decoding (33.4% vs 31.3%), suggesting a modest benefit to
   representational flexibility.

4. **Integrated code**: Context is perfectly decodable (100%) and position is
   partially decodable (33%) from the same hidden units, suggesting an
   integrated code where both context and position are jointly represented.

---

## Scope and Limitations

- This is a **computational analogy study**, not a claim about real hippocampal
  tissue.
- The core remapping effect was first demonstrated in prior 3D deep-RL work;
  this project tests whether it generalizes to a 2D grid-maze setting.
- Results are sensitive to hyperparameters, the number of contexts, and the
  random seed. Conclusions should be drawn from multiple seeds and
  configurations.
- Small network (MLP) and limited maze variety (3 layouts) may constrain the
  generality of findings.
- The one-hot context cue makes context decoding trivially easy (100%); a
  weaker context signal would be a more stringent test.

---

## Dependencies

| Library | Type | Purpose | License |
|---------|------|---------|---------|
| Eigen 3.4.0 | Header-only C++ | Matrix operations (MLP, decoder) | MPL2 |
| Catch2 v2.13.10 | Header-only C++ | Unit testing framework | BSL-1.0 |
| SQLite3 | System library | Data persistence | Public domain |

**No Python. No NumPy, pandas, scikit-learn, or matplotlib.**

Eigen and Catch2 are fetched automatically on first build (via git clone and
wget respectively). SQLite3 must be installed on the system
(`apt install libsqlite3-dev` or equivalent).

---

## Reproducibility

- Fixed random seed (default: 42)
- All configuration logged to `config.txt` and `experiments` table
- Network weights saved to `weights.bin`
- SQLite database contains full activation and episode history
- Same seed + config → same qualitative results (floating-point differences
  expected across platforms)
