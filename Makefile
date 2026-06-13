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
	@echo "📥 Downloading verified production datasets from Google Drive mirror..."
	# Source Folder Link: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
	# Using gdown to securely ingest datasets directly from the shared benchmark drive
	gdown --id 1wN_r98w9mHIsHogDsk7tSnt7Y0ImsyN8 -O data/weather.csv --no-cookies
	gdown --id 1nuMUIADOc1BNN-uDO2N7HreYl7q6qf8H -O data/exchange_rate.csv --no-cookies
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
	PYTHONPATH=. python benchmark.py
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
