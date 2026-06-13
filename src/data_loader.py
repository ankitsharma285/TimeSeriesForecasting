import os
import pandas as pd
import numpy as np

class TimeSeriesDatasetLoader:
    """
    Handles robust data loading, three-way split management (Train/Val/Test),
    and leakage-free scaling for standard time-series files.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_and_preprocess(self, filename: str, train_ratio: float = 0.7, val_ratio: float = 0.1):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing dataset: {file_path}. Please place your data files "
                f"inside the '{self.data_dir}/' folder."
            )

        # 1. Parse raw dataframe
        df = pd.read_csv(file_path)
        
        # 2. Isolate numeric features (case-insensitive date column dropping)
        columns_to_drop = [col for col in df.columns if col.lower() == 'date']
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            
        # Clean up any lingering non-numeric artifacts or empty trailing columns
        df = df.select_dtypes(include=[np.number])
        
        data_matrix = df.values.astype(np.float32)

        # 3. Time-Series Chronological Splits
        num_samples = len(data_matrix)
        if num_samples == 0:
            raise ValueError(f"The loaded dataset matrix from {filename} is completely empty.")

        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + val_ratio))

        train_data = data_matrix[:train_end]
        val_data = data_matrix[train_end:val_end]
        test_data = data_matrix[val_end:]

        # 🚨 Defensive Guard: Verify splits actually contain arrays
        if len(train_data) == 0:
            raise ValueError(
                f"Calculated training split contains 0 samples. Total rows: {num_samples}, "
                f"Train Ratio: {train_ratio}. Verify data source or ratios."
            )

        # 4. Standard Scaler calculated STRICTLY from Training Data to avoid leakage
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)
        
        # Guard against zero-variance features (constant values) to avoid divide-by-zero anomalies
        std = np.where(std == 0, 1e-5, std)

        # Normalize all splits using training scales
        norm_train = (train_data - mean) / std
        norm_val = (val_data - mean) / std
        norm_test = (test_data - mean) / std

        return norm_train, norm_val, norm_test, mean, std
