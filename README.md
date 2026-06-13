# 📊 Real-Time Time-Series Forecasting: Operational Robustness Benchmark

## Project Overview & Motivation

Time-series forecasting models are typically evaluated on clean, offline benchmark datasets. While useful for measuring predictive accuracy, these evaluations often fail to capture the conditions encountered in real-world deployment environments.

In production systems such as financial data pipelines, sensor networks, smart infrastructure, and industrial monitoring platforms, data is rarely perfect. Missing observations, noisy measurements, transient outages, anomalous spikes, and distribution drift can significantly impact forecasting performance.

This project investigates a simple question:

**How robust are forecasting models when the assumptions of clean offline evaluation no longer hold?**

To answer this, I built an end-to-end forecasting benchmark framework that combines:

* Offline model training
* Streaming inference simulation
* Operational fault injection
* Robustness evaluation
* Automated result aggregation and reporting

The framework evaluates forecasting models under a controlled set of realistic operational failures using an Operational Fault Injection Matrix that introduces:

* Missing observations
* Gaussian noise
* Spike anomalies
* Distribution drift

Rather than focusing solely on forecasting accuracy, the benchmark measures how model performance degrades as operational conditions deteriorate, providing insight into the reliability of forecasting systems in production environments.

## Core Objective

The primary objective of this project is to bridge the gap between offline forecasting evaluation and real-world operational deployment.

While forecasting models are typically trained and evaluated on clean historical datasets, production systems must operate under imperfect conditions where observations may be missing, corrupted, delayed, or subject to distribution shifts. As a result, predictive accuracy measured on offline benchmarks may not reflect real-world performance.

To investigate this gap, I developed an end-to-end robustness benchmarking framework that combines:

* Offline model training
* Streaming inference simulation
* Operational fault injection
* Automated robustness evaluation

Using a controlled Operational Fault Injection Matrix, the framework evaluates forecasting models under varying levels of missing data, Gaussian noise, spike anomalies, and distribution drift.

Rather than focusing solely on forecasting accuracy, the goal is to measure how model performance degrades as operational conditions deteriorate, identify dominant failure modes, and compare the robustness characteristics of neural forecasting architectures against simple yet competitive baseline methods.

## Defining Success

This project was not designed to maximize forecasting accuracy on a particular benchmark. Instead, the objective was to build a complete, reproducible, and extensible framework for evaluating forecasting robustness under realistic deployment conditions.

Success was defined across five dimensions:

### 1. Reproducibility

A user should be able to clone the repository, execute a small number of commands, and reproduce all benchmark results.

Success Criteria:

* Deterministic experiment configuration
* Automated evaluation pipeline
* Reproducible result generation
* Centralized configuration of datasets, models, and corruption scenarios

### 2. End-to-End Functionality

The framework should support the complete forecasting workflow rather than a standalone model implementation.

Success Criteria:

* Dataset ingestion
* Model training
* Operational fault injection
* Streaming inference simulation
* Evaluation and reporting

### 3. Robustness Evaluation

The system should quantify model behavior under realistic operational failures.

Success Criteria:

* Missing-observation simulation
* Gaussian noise injection
* Spike anomaly injection
* Distribution drift simulation
* Multiple severity levels for each fault type

### 4. Comparative Analysis

The framework should enable meaningful comparisons between forecasting approaches.

Success Criteria:

* Learned models and classical baselines evaluated under identical conditions
* Consistent metrics across datasets and corruption scenarios
* Automated result aggregation
* Quantitative degradation analysis through tables and visualizations

### 5. Insight Generation

The project should produce actionable conclusions beyond raw forecasting metrics.

Success Criteria:

* Identification of dominant failure modes
* Measurement of robustness degradation
* Analysis of model ranking changes under corruption
* Evaluation of robustness versus clean-data performance
* Practical observations relevant to production forecasting systems

### Outcome

The project is considered successful if it produces reproducible evidence about how forecasting models behave under realistic deployment conditions while remaining easy to run, extend, and analyze.

Rather than asking *"Which model is most accurate?"*, this benchmark is designed to answer *"Which models remain reliable when real-world assumptions begin to fail?"*


# 🏗️ System Architecture & Supported Configurations

The benchmark consists of two primary components:

1. A forecasting model evaluation layer
2. An Operational Fault Injection Matrix used to simulate realistic deployment failures

## Model Matrix

The framework evaluates both learned forecasting architectures and simple baseline methods to understand how model performance changes under operational stress.

### DLinear (`dlinear`)

A decomposition-based forecasting architecture that separates input sequences into trend and seasonal components before applying independent linear projections. DLinear serves as the primary neural forecasting model evaluated in this benchmark.

### Linear (`linear_v2`)

A direct linear forecasting model that maps historical observations to future horizons without explicit trend-seasonal decomposition. This model provides a lightweight neural baseline for comparison.

### Naive Persistence (`naive_persistence`)

A classical forecasting baseline that assumes the most recently observed value will persist into the future.

[X_{t+h} = X_t]

Despite its simplicity, persistence is often highly competitive on slowly changing time-series and serves as an important reference point.

### Window Repeat (`window_repeat`)

A historical-pattern baseline that repeats the most recent context window directly into the forecast horizon. This approach evaluates whether recurring local patterns alone can provide useful forecasts.

---

## Operational Fault Injection Matrix

Forecasting systems deployed in production rarely operate on perfect data. To evaluate robustness under realistic deployment conditions, the benchmark injects controlled operational faults into the input stream before inference.

Each fault is evaluated across multiple severity levels, allowing robustness degradation to be quantified and compared across forecasting models.

### Clean Baseline (`none`)

No corruption is applied to the input sequence. This scenario establishes the reference performance for each forecasting model.

### Gaussian Noise (`gaussian`)

Random high-frequency noise is added to the input sequence to simulate sensor jitter, measurement uncertainty, and noisy observations.

### Spike Anomalies (`spike`)

Transient high-magnitude perturbations are injected into the input stream to simulate outliers, abnormal events, or sudden operational disruptions.

### Distribution Drift (`drift`)

Gradual shifts are introduced into the data distribution to emulate changing environmental conditions, sensor recalibration effects, or evolving market behavior.

### Missing Observations (`missing`)

Contiguous blocks of observations are removed from the input sequence to simulate network outages, telemetry loss, delayed data arrival, or incomplete sensor reporting.

---

Together, the forecasting models and Operational Fault Injection Matrix enable systematic evaluation of forecasting robustness under realistic deployment conditions, providing insight into how predictive performance degrades as operational environments become increasingly imperfect.


