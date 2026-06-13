# ==========================================
# 📊 PRODUCTION FORECASTING BENCHMARK ENGINE
# ==========================================

.PHONY: setup train benchmark clean help

# Default task when running 'make' with no arguments
help:
	@echo "Available commands:"
	@echo "  make setup     - Install dependencies and ingest raw benchmark datasets"
	@echo "  make train     - Train neural models (dlinear, linear_v2) on clean data"
	@echo "  make benchmark - Run the full robust evaluation matrix across all faults"
	@echo "  make clean     - Flush old JSON results and checkpoints"

# Task 1: Environment Setup & Data Ingestion
setup:
	@echo "⚙️ Initializing workspace dependencies..."
	pip install -r requirements.txt gdown
	@echo "📂 Constructing isolated data directory scaffolding..."
	mkdir -p data
	@echo "📥 Downloading verified production datasets from public Google Drive folder..."
	# Downloads the folder contents directly into the data directory
	gdown --folder 1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy -O data/ --remaining-ok
	@echo "✅ Infrastructure and dataset layers are fully staged!"

# Task 2: Offline Model Training (Production Run)
train:
	@echo "🔥 Starting high-capacity baseline neural model training (20 Epochs)..."
	PYTHONPATH=. python train.py --model linear_v2 --dataset weather.csv --epochs 20 --patience 3
	PYTHONPATH=. python train.py --model dlinear --dataset weather.csv --epochs 20 --patience 3
	PYTHONPATH=. python train.py --model linear_v2 --dataset exchange_rate.csv --epochs 20 --patience 3
	PYTHONPATH=. python train.py --model dlinear --dataset exchange_rate.csv --epochs 20 --patience 3
	@echo "💾 Production training phase completed. Optimized weights saved."

# Task 3: Streaming Simulation & Multi-Variable Fault Ingestion
benchmark:
	@echo "🧪 Launching streaming adversarial operational benchmark..."
	# Running evaluations on weather dataset with multiple models and configurations
	PYTHONPATH=. python benchmark.py --model linear_v2 --dataset weather.csv --noise drift --severity 2
	PYTHONPATH=. python benchmark.py --model dlinear --dataset weather.csv --noise drift --severity 2
	# Running evaluations on exchange rate dataset
	PYTHONPATH=. python benchmark.py --model linear_v2 --dataset exchange_rate.csv --noise spike --severity 3
	PYTHONPATH=. python benchmark.py --model dlinear --dataset exchange_rate.csv --noise spike --severity 3
	@echo "📊 Aggregating multi-variable result vectors..."
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
