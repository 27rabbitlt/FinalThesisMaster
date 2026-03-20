# Convenience wrapper around cmake build

BUILD_DIR := build

.PHONY: build clean run test test-random demo viz viz-all help

## Build the solver
build:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -S . > /dev/null 2>&1
	@cmake --build $(BUILD_DIR)

## Remove build artifacts
clean:
	@rm -rf $(BUILD_DIR)

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
	@$(BUILD_DIR)/test_correctness

## Build and run the random gap search (100k trials, n=4)
test-random: build
	@$(BUILD_DIR)/test_random

## Visualize a graph (interactive HTML): make viz F=examples/two_circles_k3.json
viz: build
	@python3 visualize.py $(F) --run-solver

## Visualize all examples (tabbed HTML): make viz-all
viz-all: build
	@python3 visualize.py examples/*.json --run-solver

## Show this help
help:
	@echo "Usage:"
	@echo "  make build        Build the solver"
	@echo "  make run F=<file> Run solver on a JSON input file"
	@echo "  make demo         Run solver on all examples"
	@echo "  make test         Correctness test (DP vs brute-force)"
	@echo "  make test-random  Random instance gap search"
	@echo "  make viz F=<file> Visualize a graph (interactive HTML)"
	@echo "  make viz-all      Visualize all examples (tabbed HTML)"
	@echo "  make clean        Remove build artifacts"
