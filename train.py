import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.data_loader import TimeSeriesDatasetLoader
from src.models import get_model

def create_windows(data: np.ndarray, seq_len: int, pred_len: int):
    """Converts a continuous time-series matrix into lookback windows and targets."""
    x, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])
    return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Offline Neural Model Training with Validation")
    parser.add_argument("--model", type=str, required=True, choices=["linear_v2", "dlinear"])
    parser.add_argument("--dataset", type=str, required=True, choices=["weather.csv", "exchange_rate.csv"])
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience epochs")
    args = parser.parse_args()

    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 2. Load dataset splits using the updated 3-way loader
    data_manager = TimeSeriesDatasetLoader(data_dir="data")
    norm_train, norm_val, _, _, _ = data_manager.load_and_preprocess(args.dataset)
    
    num_features = norm_train.shape[1]
    seq_len, pred_len = 96, 24

    # 3. Form windowed loaders for both Train and Validation
    print(f"📦 Preparing window sequences...")
    x_train, y_train = create_windows(norm_train, seq_len, pred_len)
    x_val, y_val = create_windows(norm_val, seq_len, pred_len)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    # 4. Instantiate Model from Registry
    model = get_model(
        model_name=args.model,
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=num_features,
        individual=True
    ).to(device)

    # 5. Set up Optimization Engine
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. Tracking setups for saving the best model
    os.makedirs("checkpoints", exist_ok=True)
    clean_dataset_name = args.dataset.split('.')[0]
    checkpoint_path = f"checkpoints/{args.model}_{clean_dataset_name}.pt"
    
    best_val_loss = float('inf')
    patience_counter = 0

    # 7. Training & Validation Loop
    print(f"🔥 Training {args.model} on {args.dataset} with validation monitoring...")
    
    for epoch in range(args.epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        # --- SAVE BEST MODEL HOOK (Early Stopping Verification) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🌟 Validation improved! Saved best model weights to: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping triggered. Training halted at epoch {epoch+1}.")
                break
                
    print(f"✅ Training process concluded. Best Validation Loss: {best_val_loss:.6f}\n")

if __name__ == "__main__":
    main()