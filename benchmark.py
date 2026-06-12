import argparse
import json
import os
import numpy as np
import torch
from typing import Dict, Any

from src.data_loader import TimeSeriesDatasetLoader
from src.simulator import StreamingSimulator, NoiseInjector
from src.models import get_model

def run_benchmark():
    # 1. Production CLI Argument Parser
    parser = argparse.ArgumentParser(
        description="Streaming Forecasting Benchmark: Robustness Under Drift & Noise"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["weather.csv", "exchange_rate.csv"],
        help="Target dataset file located inside the data/ directory."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["dlinear", "linear_v2", "naive_persistence", "window_repeat"],
        help="Forecasting model architecture to evaluate."
    )
    parser.add_argument(
        "--noise", 
        type=str, 
        default="none", 
        choices=["none", "gaussian", "missing", "spike", "drift"],
        help="Type of data corruption/drift to inject into the streaming lookback window."
    )
    parser.add_argument(
        "--severity", 
        type=float, 
        default=0.0,
        help="Intensity parameter for noise injection, normalized from 0.0 to 1.0."
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints",
        help="Directory where trained model weights (.pt) are stored."
    )
    args = parser.parse_args()

    # 2. Setup Compute Device Context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ℹ️ Using execution device context: {device}")

    # 3. Initialize Data Loader Pipeline
    data_manager = TimeSeriesDatasetLoader(data_dir="data")
    try:
        # Secure chronological three-way data separation
        norm_train, norm_val, norm_test, mean, std = data_manager.load_and_preprocess(args.dataset)
        
    except FileNotFoundError as e:
        print(f"❌ Execution Error: {e}")
        return

    num_features = norm_test.shape[1]
    seq_len = 96
    pred_len = 24

    # 4. Instantiate streaming simulator components
    # Ensure the simulator runs EXCLUSIVELY over the unseen test data partition
    simulator = StreamingSimulator(norm_test, seq_len=seq_len, pred_len=pred_len)
    injector = NoiseInjector(seed=42)

    # 5. Dynamic Model Resolution via Registry Factory
    model = get_model(
        model_name=args.model,
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=num_features,
        individual=True
    )

    # 6. Leakage-free Weight Initialization (Only neural models require checkpoints)
    is_neural_model = "naive" not in args.model and "window" not in args.model
    if is_neural_model:
        checkpoint_name = f"{args.model}_{args.dataset.split('.')[0]}.pt"
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            print(f"🔒 Loading optimized model weights from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}. Running with random initialization.")
    else:
        print(f"📊 Running zero-parameter statistical baseline. Zero training weights required.")

    # Move model cleanly to target runtime architecture context
    model = model.to(device)
    model.eval()

    # 7. Execute Streaming Simulation Loop
    maes, mses = [], []
    print(f"⏳ Processing production stream simulation loop...")

    for x_clean, y_true in simulator.stream_windows():
        # Inject on-the-fly production corruption into historical lookback sequence
        x_corrupted = injector.inject(x_clean, args.noise, args.severity)
        
        # Convert inputs to batched Torch tensors [Batch=1, Seq_len, Channels] and target context
        x_tensor = torch.tensor(x_corrupted, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Isolated forward evaluation pass
        with torch.no_grad():
            y_pred_tensor = model(x_tensor)
            # FIX: Explicit batch index slice extraction [0] rather than ambiguous squeeze calls
            # to guarantee safe multidimensional output extraction under single feature datasets
            y_pred = y_pred_tensor[0].cpu().numpy()
            
        # Evaluate operational performance metrics against clean ground-truth targets
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        
        maes.append(mae)
        mses.append(mse)

    # 8. Compile Summary Performance Payload
    summary_results = {
        "dataset": args.dataset,
        "model": args.model,
        "noise_type": args.noise,
        "severity": args.severity,
        "mean_mae": float(np.mean(maes)),
        "mean_mse": float(np.mean(mses)),
        "total_windows_evaluated": len(maes)
    }

    # 9. Structured Serialization to Results Store
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate structured clean filenames for the report compiler
    dataset_prefix = args.dataset.split('.')[0]
    out_filename = f"{dataset_prefix}_{args.model}_{args.noise}_{args.severity}.json"
    out_path = os.path.join(out_dir, out_filename)
    
    with open(out_path, "w") as f:
        json.dump(summary_results, f, indent=4)
        
    print(f"✅ Success! Benchmark run finished completely.")
    print(f"📝 Report serialized to: {out_path}")
    print(f"📊 Results Summary -> MAE: {summary_results['mean_mae']:.4f} | MSE: {summary_results['mean_mse']:.4f}\n")

if __name__ == "__main__":
    run_benchmark()