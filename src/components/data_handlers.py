import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod

class BaseDataHandler(ABC):
    def __init__(self, data_path: str, input_column: str, document_column: str, gt_column: str = None,
                 sample_size: int = 1000, random_seed: int = 42, **kwargs):
        self.data_path = data_path
        self.input_column = input_column
        self.document_column = document_column
        self.gt_column = gt_column
        self.sample_size = sample_size
        self.random_seed = random_seed

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    # TODO maybe not force everyone to use dataframes
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    def sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")
        if self.sample_size is None or len(df) <= self.sample_size:
            print(f"DataFrame has {len(df)} rows. Sample size is {self.sample_size}. Using the entire DataFrame or as is.")
            return df
        
        np.random.seed(self.random_seed)
        sampled_df = df.sample(n=self.sample_size, random_state=self.random_seed)
        print(f"Sampled {len(sampled_df)} rows from {len(df)} total rows using seed {self.random_seed}.")
        return sampled_df

class JsonlDataHandler(BaseDataHandler):
    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_json(self.data_path, lines=True)
            print(f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()} from {self.data_path}")
            # Validate required columns
            required_cols = [self.input_column, self.document_column]
            if self.gt_column:
                required_cols.append(self.gt_column)
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {self.data_path}. Available columns: {df.columns.tolist()}")
            return df
        except ValueError as e: # Handles JSON decoding errors and other pandas value errors
            print(f"Error reading JSONL file {self.data_path}: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {self.data_path}: {e}")
            raise

# TODO more datafile types (parquet, csv, sharegpt, alpaca, maybe support online grabbing from HF?)
class CSVDataHandler(BaseDataHandler):
    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError()
    
