import pandas as pd
from time import strftime
from pathlib import Path
from abc import ABC, abstractmethod
from utils.config_utils import ConfigDict
import json


class BaseDataHandler(ABC):
    def __init__(
        self,
        data_path: str | Path,
        input_column: str,
        document_column: str,
        gt_column: str = None,
        sample_size: int = 1000,
        random_seed: int = 42,
        cache_path: str | Path = "./cache",
        **kwargs,
    ):
        self.data_path = Path(data_path)
        self.input_column = input_column
        self.document_column = document_column
        self.gt_column = gt_column
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.cache_path = Path(cache_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.cache_path.mkdir(exist_ok=True, parents=True)

    # TODO maybe not force everyone to use dataframes
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_cache(self, model_answers: list[str], model_config: ConfigDict):
        pass

    @abstractmethod
    def assemble_cache_filepath(self, model_config: ConfigDict) -> Path:
        pass

    @abstractmethod
    def save_final_output(self, evaluations: list[dict], run_config: ConfigDict, model_config: ConfigDict):
        pass

    def sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")
        if self.sample_size is None or len(df) <= self.sample_size:
            print(
                f"DataFrame has {len(df)} rows. Sample size is {self.sample_size}. Using the entire DataFrame or as is."
            )
            return df

        sampled_df = df.sample(n=self.sample_size, random_state=self.random_seed)
        print(
            f"Sampled {len(sampled_df)} rows from {len(df)} total rows using seed {self.random_seed}."
        )
        return sampled_df


class JsonlDataHandler(BaseDataHandler):
    def __init__(
        self,
        data_path: str | Path,
        input_column: str,
        document_column: str,
        gt_column: str = None,
        sample_size: int = 1000,
        random_seed: int = 42,
        cache_path: str | Path = "./cache",
        **kwargs,
    ):
        super().__init__(
            data_path,
            input_column,
            document_column,
            gt_column,
            sample_size,
            random_seed,
            cache_path,
            **kwargs,
        )
        self.df = self.load_data()
        self.sampled_df = self.sample_dataframe(self.df)

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_json(self.data_path, lines=True)
            print(
                f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()} from {self.data_path}"
            )
            # Validate required columns
            required_cols = [self.input_column, self.document_column]
            if self.gt_column:
                required_cols.append(self.gt_column)
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"Required column '{col}' not found in {self.data_path}. Available columns: {df.columns.tolist()}"
                    )
            return df
        except (
            ValueError
        ) as e:  # Handles JSON decoding errors and other pandas value errors
            print(f"Error reading JSONL file {self.data_path}: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {self.data_path}: {e}")
            raise

    def save_cache(self, model_answers: list[str], model_config: ConfigDict):
        full_response_cache_path = self.assemble_cache_filepath(model_config)

        cache_metadata = self._format_metadata(model_config)

        cache_data = {
            "metadata": cache_metadata,
            "data": [{"model_answer": answer} for answer in model_answers],
        }

        with open(full_response_cache_path, "w") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    def find_and_load_cache(self, model_config: ConfigDict) -> list[str] | None:
        response_cache_filepath = self.assemble_cache_filepath(model_config)

        if response_cache_filepath.exists():
            print(f"Loading cached model responses from {response_cache_filepath}")
            with open(response_cache_filepath) as f:
                try:
                    cache = json.load(f)
                    loaded_metadata = cache.get("metadata", {})
                    runtime_metadata = self._format_metadata(model_config)
                    if (
                        loaded_metadata.get("model_name_used")
                        == runtime_metadata["model_name_used"]
                        and loaded_metadata.get("lora_adapter_path_used")
                        == runtime_metadata["lora_adapter_path_used"]
                    ):
                        answer_dicts = cache.get("data", {})
                        model_answers = [
                            answer_dict["model_answer"] for answer_dict in answer_dicts
                        ]
                        return model_answers
                    return None
                except ValueError as ve:
                    print("Cache formatting failure: ", ve)
                    return None
                except Exception as e:
                    print("Unexpected cache load failure: ", e)
                    return None
        return None

    def assemble_cache_filepath(self, model_config: ConfigDict) -> Path:
        model_name_sanitized = Path(model_config.get("model_name_or_path")).stem
        lora_path = model_config.get("lora_adapter_path")
        adapter_name_sanitized = Path(lora_path).stem if lora_path else "no_adapter"
        data_basename = self.data_path.stem

        cache_filename = f"{model_name_sanitized}_adapter_{adapter_name_sanitized}_seed{self.random_seed}_n{self.sample_size}_{data_basename}.json"
        return Path(self.cache_path, cache_filename)

    def _format_metadata(self, model_config: ConfigDict) -> dict:
        cache_metadata = {
            "model_name_used": model_config.get("model_name_or_path"),
            "lora_adapter_path_used": model_config.get("lora_adapter_path"),
            "generation_timestamp": strftime("%Y-%m-%d %H:%M:%S"),
            "sampled_from": str(self.data_path),
            "random_seed": self.random_seed,
            "sample_size": self.sample_size,
            "cache_version": "1.0",
        }
        return cache_metadata
    
    def assemble_final_output_filepath(self, run_config: ConfigDict, model_config: ConfigDict) -> Path:
        base_results_dir = Path(run_config.get("results_dir", "../results"))
        base_results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = strftime("%Y%m%d_%H%M%S")

        model_name_sanitized = Path(model_config.get("model_name_or_path")).stem
        lora_path = model_config.get("lora_adapter_path")
        adapter_name_sanitized = Path(lora_path).stem if lora_path else "no_adapter"
        model_output_dirname = f"{model_name_sanitized}_adapter_{adapter_name_sanitized}"
        output_dir = Path(base_results_dir, model_output_dirname)
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_name_sanitized = run_config.get("metric", "unknown_metric").replace(" ", "_")
        output_fn = f"{metric_name_sanitized}_{timestamp}.json"
        return Path(output_dir, output_fn)
    
    def save_final_output(self, evaluations: list[dict], run_config: ConfigDict, model_config: ConfigDict):
        final_metadata = self._format_metadata(model_config)
        final_output_path = self.assemble_final_output_filepath(run_config, model_config)

        final_data = {
            "metadata": final_metadata,
            "data": evaluations,
        }

        with open(final_output_path, "w") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {final_output_path}")




# TODO more datafile types (parquet, csv, sharegpt, alpaca, maybe support online grabbing from HF?)
class CSVDataHandler(BaseDataHandler):
    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError()
