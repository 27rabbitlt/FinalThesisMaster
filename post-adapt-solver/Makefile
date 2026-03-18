# Convenience wrapper around cmake build

BUILD_DIR := build

.PHONY: build clean run test test-random demo help

## Build the solver
build:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -S . > /dev/null 2>&1
	@cmake --build $(BUILD_DIR)

## Remove build artifacts
clean:
	@rm -rf $(BUILD_DIR) test_random

## Run solver on a JSON file: make run F=examples/gap_demo.json
run: build
	@$(BUILD_DIR)/solver $(F)

## Run all example inputs
demo: build
	@for f in examples/*.json; do \
		echo "=== $$f ==="; \
		$(BUILD_DIR)/solver "$$f"; \
		echo; \
	done

## Correctness test: DP vs brute-force on random instances
test: build
	@g++ -O2 -std=c++17 -o test_correctness test_correctness.cpp
	@./test_correctness

## Build and run the random gap search (100k trials, n=4)
test-random: build
	@g++ -O2 -std=c++17 -o test_random test_random.cpp
	@./test_random

## Show this help
help:
	@echo "Usage:"
	@echo "  make build        Build the solver"
	@echo "  make run F=<file> Run solver on a JSON input file"
	@echo "  make demo         Run solver on all examples"
	@echo "  make test         Correctness test (DP vs brute-force)"
	@echo "  make test-random  Random instance gap search"
	@echo "  make clean        Remove build artifacts"
