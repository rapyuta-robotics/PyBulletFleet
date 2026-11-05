# PyBulletFleet Development Makefile
# Usage: make help

.DEFAULT_GOAL := help

.PHONY: help lint format typecheck test test-fast verify docs bench-smoke bench-full bench-release rmf-demos-setup ros-native-setup rmf-matrix clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:[^#]*## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint:  ## Run all pre-commit hooks (black, pyright, flake8)
	pre-commit run --all-files --show-diff-on-failure

format:  ## Auto-format code with black
	black pybullet_fleet tests examples benchmark docs/conf.py

typecheck:  ## Run pyright type checker
	pyright

test:  ## Run tests with coverage (CI test job equivalent)
	pytest tests/ -v --tb=short --cov=pybullet_fleet --cov-report=term-missing --cov-fail-under=75

test-fast:  ## Quick test run (stop on first failure)
	pytest tests/ -x -q --no-cov

test-clean-install:  ## Build wheel + install in a clean venv + smoke test (catches packaging/data-file bugs)
	bash scripts/test_clean_install.sh

verify: lint test  ## Full verification (lint + test, CI subset)

docs:  ## Build documentation (warnings as errors)
	cd docs && sphinx-build -E -a -W -b html . _build/html

bench-smoke:  ## Quick benchmark (~10 seconds)
	python benchmark/run_benchmark.py --duration 10

bench-full:  ## Full benchmark sweep
	python benchmark/run_benchmark.py --sweep 100 500 1000

bench-steady:  ## Steady-state scenes (fleet keeps moving): 1000 robots sparse/dense/carrying/idle, 200 in a warehouse
	python benchmark/run_benchmark.py --type steady --compare sparse dense carrying idle --agents 1000 --repetitions 3
	python benchmark/run_benchmark.py --type steady --scenario warehouse --agents 200 --repetitions 3

bench-release:  ## Release benchmark sweep for docs refresh
	python benchmark/run_benchmark.py --sweep 100 250 500 1000 2000 --duration 10 --repetitions 3
	python benchmark/run_benchmark.py --type mobile_control_path --controller per_agent batch --command-interface per_agent fleet --sweep 100 500 1000 --steps 600 --repetitions 3

rmf-demos-setup:  ## Build native Open-RMF demos workspace for Jazzy
	bash scripts/setup_rmf_demos_jazzy.sh

ros-native-setup:  ## Build native ROS 2 Jazzy overlay workspace
	bash scripts/setup_native_ros2_jazzy.sh

rmf-matrix:  ## Run native RMF client-mode matrix (requires source .ros2_native_env)
	bash docker/test_rmf_client_modes.sh

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
