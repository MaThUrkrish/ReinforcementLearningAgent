# remap2d - Global Remapping in a Simplified 2D RL Agent
# Build system: plain GNU Make (no CMake dependency)
# All code is C++17. No Python anywhere.

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc -Ithird_party/eigen_src -Ithird_party
LDFLAGS  = -lsqlite3 -lm
TESTFLAGS = -DCATCH_CONFIG_MAIN

# Core source files (shared by all binaries)
SRC_CORE = src/MazeEnvironment.cpp src/DQNAgent.cpp src/ReplayBuffer.cpp \
           src/RunLogger.cpp src/utils/Random.cpp

# Analysis source files
SRC_ANALYSIS = src/analysis/RunLoader.cpp src/analysis/RateMaps.cpp \
               src/analysis/Correlation.cpp src/analysis/Decoder.cpp \
               src/analysis/Report.cpp src/utils/Svg.cpp

HEADERS = src/Config.h src/MazeEnvironment.h src/DQNAgent.h src/ReplayBuffer.h \
          src/RunLogger.h src/utils/Random.h src/utils/Svg.h \
          src/analysis/RunLoader.h src/analysis/RateMaps.h \
          src/analysis/Correlation.h src/analysis/Decoder.h \
          src/analysis/Report.h

# Training binary
train_agent: src/main.cpp $(SRC_CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o train_agent src/main.cpp $(SRC_CORE) $(LDFLAGS)

# Analysis binary
analyze: src/analyze_main.cpp $(SRC_ANALYSIS) $(SRC_CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o analyze src/analyze_main.cpp $(SRC_ANALYSIS) $(SRC_CORE) $(LDFLAGS)

# Unit test binary (Catch2 v2 single header)
test_remap2d: tests/test_main.cpp $(SRC_CORE) $(SRC_ANALYSIS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o test_remap2d tests/test_main.cpp $(SRC_CORE) $(SRC_ANALYSIS) $(LDFLAGS)

test: test_remap2d
	./test_remap2d

# Run the full reference experiment (replay on + off) and analyze both
experiment: train_agent analyze
	./train_agent --contexts 0,1,0 --episodes-per-context 300 --hidden-size 64 --replay on  --seed 42 --out runs/ref_replay_on
	./train_agent --contexts 0,1,0 --episodes-per-context 300 --hidden-size 64 --replay off --seed 42 --out runs/ref_replay_off
	./analyze --run runs/ref_replay_on
	./analyze --run runs/ref_replay_off

clean:
	rm -f train_agent analyze test_remap2d
	rm -rf runs/ref_replay_on runs/ref_replay_off

.PHONY: test experiment clean
