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
# Task 1: Environment Setup & Data Ingestion
setup:
	@echo "⚙️ Initializing workspace dependencies..."
	pip install -r requirements.txt
	@echo "📂 Constructing isolated data directory scaffolding..."
	mkdir -p data
	@echo "📥 Downloading verified production datasets from benchmark mirrors..."
	# Downloads the exact pre-processed weather dataset used in the DLinear paper
	curl -L -o data/weather.csv "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/weather/weather.csv"
	# Downloads the exact pre-processed exchange rate dataset
	curl -L -o data/exchange_rate.csv "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/exchange_rate/exchange_rate.csv"
	@echo "✅ Infrastructure and dataset layers are fully staged!"

