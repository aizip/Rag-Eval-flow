from pathlib import Path
import pandas as pd
import json
import re


def sample_dataframe(
    sample_size: int, df: pd.DataFrame, random_seed: int = 42
) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        print(
            f"DataFrame has {len(df)} rows. Sample size is {sample_size}. Using the entire DataFrame or as is."
        )
        return df

    sampled_df = df.sample(n=sample_size, random_state=random_seed).sort_index()
    print(
        f"Sampled {len(sampled_df)} rows from {len(df)} total rows using seed {random_seed}."
    )
    return sampled_df


def _parse_filename_fallback(filename_stem: str) -> tuple[str | None, str | None]:
    """
    Parses a filename stem to extract a metric name and a formatted timestamp.

    This is the fallback for legacy files named like:
    'metric_name_may_be_multiple_words_20250812_144500.json'
    """
    match = re.match(r"^(.+)_(\d{8})_(\d{6})$", filename_stem)

    if not match:
        return None, None

    metric_name = match.group(1)
    date_str = match.group(2)  # "YYYYMMDD"
    time_str = match.group(3)  # "HHMMSS"

    # Format into a standard, comparable timestamp string
    timestamp = (
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
        f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    )

    return metric_name, timestamp


def find_latest_metric_files(directory_path: str | Path) -> list[Path]:
    """
    Finds the most recent file for each unique metric in a directory.

    This function prioritizes reading a 'metadata' object from each .json file.
    If the 'metadata' object or its 'generation_timestamp' is missing, it
    falls back to parsing the filename for the metric name and timestamp.

    Args:
        directory_path: The path to the directory containing the metric files.

    Returns:
        A list of Path objects, one for each unique metric, corresponding to the
        most recent timestamp found.
    """
    latest_files = {}
    target_dir = Path(directory_path)

    if not target_dir.is_dir():
        return []

    for file_path in target_dir.glob("*.json"):
        metric_name, timestamp = None, None

        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)

            metadata = content.get("metadata")
            if metadata:
                timestamp = metadata.get("generation_timestamp")
                metric_name = metadata.get("metric_name")

        except (IOError, json.JSONDecodeError):
            # If file is unreadable or not valid JSON, we will proceed to the
            # fallback method below, as 'timestamp' will still be None.
            pass

        if not timestamp:
            metric_name, timestamp = _parse_filename_fallback(file_path.stem)
        # If metadata gave us a timestamp but not a name, infer the name.
        elif not metric_name:
            inferred_name, _ = _parse_filename_fallback(file_path.stem)
            metric_name = inferred_name if inferred_name else file_path.stem

        # Collate
        if metric_name and timestamp:
            current_latest_ts = latest_files.get(metric_name, {}).get("timestamp")

            if not current_latest_ts or timestamp > current_latest_ts:
                latest_files[metric_name] = {"path": file_path, "timestamp": timestamp}

    return [data["path"] for data in latest_files.values()]


def construct_one_model_metrics_df(
    filepath_list: list[Path],
) -> tuple[pd.DataFrame, set]:
    """
    Concatenates all the metrics from one model/adapter columnwise on the same input data

    Assumes the data schema used in main.py

    Args:
        filepath_list: A list of metric .json(l) filenames to concatenate

    Returns:
        A single DataFrame with sample_size rows and all metrics and response data columns
        and a set strings of the discovered metric names
    """
    if not filepath_list:
        print("WARNING: Metric Collation found no evaluation outputs.")
        return pd.DataFrame(), set()

    discovered_metrics = set()
    individual_metric_dfs = []
    # TODO assemble data portion of the report, requires seed, sample_size, data_path.
    data_extraction_example = filepath_list[0]

    try:
        with open(data_extraction_example) as f:
            example = json.load(f)
            metadata = example.get("metadata", {})
            model_answers = pd.Series(
                example.get("model_answer", []), name="model_answer"
            )

        data_path = Path(metadata["sampled_from"])
        random_seed = metadata["random_seed"]
        sample_size = metadata["sample_size"]

    except Exception as e:
        df = pd.read_json(data_extraction_example, lines=True)
        row = df.iloc[0, :]
        data_path = Path(row["data_path"])
        random_seed = row["random_seed"]
        sample_size = row["sample_size"]
        model_answers = df["model_answer"]

    data_df = pd.read_json(data_path, lines=True)
    metadata_df = pd.json_normalize(data_df["metadata"])
    full_df = pd.concat([data_df, metadata_df], axis=1)
    
    sampled_data_df = sample_dataframe(
        sample_size, full_df, random_seed
    ).sort_index()
    sampled_data_df["orig_index"] = sampled_data_df.index.to_list()
    sampled_data_df.reset_index(inplace=True)

    final_data_df = pd.concat(
        [sampled_data_df, model_answers], axis=1
    )
    final_data_df.reset_index(inplace=True)
    individual_metric_dfs.append(final_data_df)

    for file_path in filepath_list:
        metric_df, metric_name = construct_one_metric_df(file_path)
        discovered_metrics.add(metric_name)
        individual_metric_dfs.append(metric_df)

    return pd.concat(individual_metric_dfs, axis=1), discovered_metrics


def construct_one_metric_df(file_path: Path) -> tuple[pd.DataFrame, str]:
    try:
        with open(file_path) as f:
            all_data = json.load(f)
            metadata = all_data.get("metadata", {})
            data = all_data.get("data", [])

        data_df = pd.DataFrame(data)[["score", "explanation"]]
        if "metric_name" not in metadata.keys():
            metric_name, _ = _parse_filename_fallback(file_path.stem)
        else:
            metric_name = metadata["metric_name"]

        return_slice = data_df.rename(
            columns={
                "score": f"{metric_name}_score",
                "explanation": f"{metric_name}_explanation",
            }
        )
        return return_slice, metric_name

    except Exception as e:
        return_slice, metric_name = _legacy_construct_one_metric_df(file_path)
        return return_slice, metric_name


def _legacy_construct_one_metric_df(file_path: Path) -> tuple[pd.DataFrame, str]:
    timestamp_len = 15
    # Set schema
    final_df = pd.read_json(file_path, lines=True)

    base_name = file_path.stem
    metric_name = base_name[: -(timestamp_len + 1)]

    ev_data = final_df["evaluation"].apply(pd.Series)[["score", "explanation"]]
    renamed_ev_data = ev_data.rename(
        columns={
            "score": f"{metric_name}_score",
            "explanation": f"{metric_name}_explanation",
        }
    )

    # final_df = pd.concat([final_df.drop(columns=['evaluation']), renamed_ev_data], axis=1)

    return renamed_ev_data, metric_name


def compute_rejection_rates(
    fully_assembled_df: pd.DataFrame, return_df: bool = False
) -> tuple[float, float] | dict[str, pd.DataFrame]:
    """Calculates refusal detection precision and negative predictive value.

    Args:
        fully_assembled_df (pd.DataFrame): DataFrame with predictions and labels.
            Must contain:
            - 'refusal_presence_score' (int/float): 1 for predicted refusal, 0 otherwise.
            - 'response_type' (str): Ground truth label. 'refusal' or 'followup'
            are treated as positive cases.
        return_df (bool, optional): If True, returns a dictionary of DataFrames
            for each confusion matrix category. Defaults to False.

    Returns:
        A tuple containing (precision, negative_predictive_value), or a
        dictionary of DataFrames if `return_df` is True.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    if not (
        "refusal_presence_score" in fully_assembled_df.columns
        and "response_type" in fully_assembled_df.columns
    ):
        raise ValueError(
            "Requires both refusal_presence metric AND a metadata response_type to be run."
        )

    tp_df = fully_assembled_df[
        (fully_assembled_df["refusal_presence_score"] == 1)
        & (
            (fully_assembled_df["response_type"] == "refusal")
            | (fully_assembled_df["response_type"] == "followup")
        )
    ]

    fp_df = fully_assembled_df[
        (fully_assembled_df["refusal_presence_score"] == 1)
        & (fully_assembled_df["response_type"] == "answer")
    ]

    fn_df = fully_assembled_df[
        (fully_assembled_df["refusal_presence_score"] == 0)
        & (
            (fully_assembled_df["response_type"] == "refusal")
            | (fully_assembled_df["response_type"] == "followup")
        )
    ]

    tn_df = fully_assembled_df[
        (fully_assembled_df["refusal_presence_score"] == 0)
        & (fully_assembled_df["response_type"] == "answer")
    ]

    tp, fp, fn, tn = map(lambda x: x.shape[0], [tp_df, fp_df, fn_df, tn_df])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    negative_detection_accuracy = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return (
        (precision, negative_detection_accuracy)
        if not return_df
        else {
            "true_positives": tp_df,
            "false_positives": fp_df,
            "false_negatives": fn_df,
            "true_negatives": tn_df,
        }
    )

def assemble_accurate_df(fully_assembled_df: pd.DataFrame):
    df_dict =  compute_rejection_rates(fully_assembled_df, return_df=True)
    return pd.concat([df_dict["true_positives"], df_dict["true_negatives"]]).sort_index(ascending=True)


# TODO: update to not need the following arcane spell
def add_surrogate_key(df, keys, key_name="_merge_key"):
    import hashlib

    """
    Creates a new surrogate key column by hashing a canonical
    JSON representation of the specified key columns.
    """
    df_copy = df.copy()

    def create_hash(row):
        canonical_json = json.dumps(row.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    df_copy[key_name] = df_copy[keys].apply(create_hash, axis=1)
    return df_copy
