# ==========================================
# 📊 PRODUCTION FORECASTING BENCHMARK ENGINE
# ==========================================

.PHONY: setup train benchmark clean help

# Default task when running 'make' with no arguments
help:
	@echo "Available commands:"
	@echo "  make setup     - Install dependencies and prepare workspace"
	@echo "  make train     - Train neural models (dlinear, linear_v2) on local clean data"
	@echo "  make benchmark - Run the full robust evaluation matrix across all faults"
	@echo "  make clean     - Flush old JSON results and checkpoints"

# Task 1: Environment Setup
setup:
	@echo "⚙️ Initializing workspace dependencies..."
	pip install -r requirements.txt
	@echo "✅ Infrastructure layers are fully staged!"

# Task 2: Offline Model Training (Production Run)
train:
	@echo "🔥 Starting high-capacity baseline neural model training (20 Epochs)..."
	PYTHONPATH=. python train.py --model linear_v2 --dataset weather.csv --epochs 20 --patience 3
	PYTHONPATH=. python train.py --model dlinear --dataset weather.csv --epochs 20 --patience 3
	PYTHONPATH=. python train.py --model linear_v2 --dataset exchange_rate.csv --epochs 20 --patience 3
	PYTHONPATH=. python train.py --model dlinear --dataset exchange_rate.csv --epochs 20 --patience 3
	@echo "💾 Production training phase completed. Optimized weights saved."

# Task 3: The Step C Automation Matrix (Comprehensive Sweep)
benchmark:
	@echo "🚀 Initiating full sequential data corruption streaming sweep..."
	@mkdir -p results
	@for dataset in weather.csv exchange_rate.csv; do \
		for model in naive_persistence window_repeat linear_v2 dlinear; do \
			echo "Running benchmark for $$model on $$dataset (Clean Base)..."; \
			PYTHONPATH=. python benchmark.py --model $$model --dataset $$dataset --noise none --severity 0.0; \
			for noise in gaussian missing spike drift; do \
				for severity in 0.1 0.3; do \
					echo "Running benchmark for $$model on $$dataset | Noise: $$noise | Severity: $$severity"; \
					PYTHONPATH=. python benchmark.py --model $$model --dataset $$dataset --noise $$noise --severity $$severity; \
				done \
			done \
		done \
	done
	@echo "🏁 Sweep completed. Packaging visual presentation layer..."
	PYTHONPATH=. python generate_report.py

# Task 4: System Sanitation
clean:
	@echo "🧹 Flushing cache, local metrics, and model checkpoints..."
	rm -rf results/*.json
	rm -rf results/plots/*.png
	rm -rf results/summary_matrix.md
	rm -rf checkpoints/*.pt
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✨ Workspace sanitized completely."
