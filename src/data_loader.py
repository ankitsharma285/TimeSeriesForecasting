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
        
        # 2. Isolate numeric features (drop date string column)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        
        data_matrix = df.values.astype(np.float32)

        # 3. Time-Series Chronological Splits
        num_samples = len(data_matrix)
        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + val_ratio))

        train_data = data_matrix[:train_end]
        val_data = data_matrix[train_end:val_end]
        test_data = data_matrix[val_end:]

        # 4. Standard Scaler calculated STRICTLY from Training Data
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0) + 1e-5

        # Normalize all splits using training scales
        norm_train = (train_data - mean) / std
        norm_val = (val_data - mean) / std
        norm_test = (test_data - mean) / std

        return norm_train, norm_val, norm_test, mean, std