# ==========================================
# 📊 PRODUCTION FORECASTING BENCHMARK ENGINE
# ==========================================

.PHONY: setup train benchmark clean help

# Default task when running 'make' with no arguments
help:
	@echo "Available commands:"
	@echo "  make setup     - Install required Python dependencies"
	@echo "  make train     - Train neural models (dlinear, linear_v2) on clean data"
	@echo "  make benchmark - Run the full robust evaluation matrix (Step C)"
	@echo "  make clean     - Flush old JSON results and checkpoints"

# Task 1: Environment Setup
setup:
	pip install torch numpy pandas tabulate


