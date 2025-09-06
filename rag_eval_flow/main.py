import argparse
import os
import time
import pandas as pd
from pathlib import Path
import json

from factory import ObjectFactory
from utils.config_utils import load_config, get_config_value, recursive_update
from utils.data_utils import sample_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Run model evaluation based on YAML configuration."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="../config/evaluation_config.yaml",
        help="Path to the main YAML configuration file.",
    )
    parser.add_argument(
        "--pipeline_key",
        type=str,
        required=True,
        help="Key of the evaluation pipeline to run from the YAML config.",
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to the input data file (e.g., JSONL)."
    )
    parser.add_argument(
        "--model_key",
        type=str,
        help="Key of the model to test from the YAML config (e.g., 'huggingface_causal_lm').",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the HuggingFace model to load.",
    )
    parser.add_argument(
        "--judge_model_name", type=str, help="Name of the judge model (e.g., 'gpt-4o')."
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (optional).",
    )
    parser.add_argument(
        "--metric", type=str, help="Metric to evaluate (e.g., 'response_coherence')."
    )
    parser.add_argument("--sample_size", type=int, help="Number of samples to process.")
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for model inference."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, help="Max new tokens for model generation."
    )
    parser.add_argument(
        "--output_base_dir", type=str, help="Base directory to save results."
    )
    parser.add_argument(
        "--cache_base_dir", type=str, help="Base directory for caching."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config_path)
    if not config:
        return

    pipeline_config = get_config_value(
        config, f"evaluation_pipelines.{args.pipeline_key}"
    )
    if not pipeline_config:
        print(f"Error: Pipeline key '{args.pipeline_key}' not found in configuration.")
        return

    # 2. Build Effective Run Configuration (CLI > Pipeline > Base)
    run_config = config.get("base_config", {}).copy()
    pipeline_level_settings = {
        k: v for k, v in pipeline_config.items() if not k.endswith("_config_overrides")
    }
    recursive_update(run_config, pipeline_level_settings)

    cli_general_overrides = {
        "data_path": args.data_path,
        "metric": args.metric,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "results_dir": args.output_base_dir,
        "cache_dir": args.cache_base_dir,
        "model_key": args.model_key,
    }
    recursive_update(
        run_config, {k: v for k, v in cli_general_overrides.items() if v is not None}
    )

    # 3. Instantiate Components with Finalized Configs
    factory = ObjectFactory(config)

    # Data Handler
    data_source_key = run_config.get("data_source_key")
    data_handler_config = config.get("data_sources", {}).get(data_source_key, {}).copy()
    recursive_update(
        data_handler_config,
        {
            "data_path": run_config.get("data_path"),
            "sample_size": run_config.get("sample_size"),
            "random_seed": run_config.get("random_seed"),
        },
    )
    data_handler = factory.create(
        data_source_key, "data_sources", **data_handler_config
    )
    if not data_handler:
        return

    # Model Under Test
    model_key = run_config.get("model_key")
    model_config = config.get("models", {}).get(model_key, {}).copy()
    recursive_update(model_config, pipeline_config.get("model_config_overrides", {}))
    cli_model_overrides = {
        "model_name_or_path": args.model_name_or_path,
        "lora_adapter_path": args.lora_adapter_path,
    }
    recursive_update(
        model_config, {k: v for k, v in cli_model_overrides.items() if v is not None}
    )
    recursive_update(
        model_config,
        {
            "max_new_tokens": run_config.get("max_new_tokens"),
            "batch_size": run_config.get("batch_size"),
        },
    )


    
    # Judge Model
    judge_key = run_config.get("judge_key")
    judge_config = config.get("judges", {}).get(judge_key, {}).copy()
    recursive_update(judge_config, pipeline_config.get("judge_config_overrides", {}))
    recursive_update(judge_config, {"judge_model_name": args.judge_model_name})
    model_name_for_paths = os.path.basename(
        model_config.get("model_name_or_path", "unknown_model")
    ).replace("/", "_")
    recursive_update(
        judge_config,
        {
            "metric": run_config.get("metric"),
            "base_results_dir": run_config.get("results_dir"),
            "model_name_for_paths": model_name_for_paths,
        },
    )
    judge_model = factory.create(judge_key, "judges", **judge_config)
    if not judge_model:
        return

    # 4. Run Evaluation Flow
    print("Loading and sampling data...")
    dataframe = data_handler.load_data()
    sampled_df = sample_dataframe(run_config.get("sample_size"), dataframe, run_config.get("random_seed"))
    queries, contexts = (
        sampled_df[data_handler.input_column].tolist(),
        sampled_df[data_handler.document_column].tolist(),
    )
    gt_answers = (
        sampled_df[data_handler.gt_column].tolist()
        if data_handler.gt_column in sampled_df
        else [None] * len(queries)
    )

    # Prepare cache
    response_cache_path = data_handler.assemble_cache_filepath(model_config)
    found_answers = data_handler.find_and_load_cache(model_config)
    
    if found_answers:
        model_answers = found_answers
    else:
        model_under_test = factory.create(model_key, "models", **model_config)
        if not model_under_test:
            return
        print(
            f"Generating answers using model: {model_config.get('model_name_or_path')}"
        )
        model_answers = model_under_test.generate_answers(queries, contexts)
        data_handler.save_cache(model_answers, model_config)

    print("Sending data to judge model for evaluation...")
    evaluations = judge_model.evaluate(
        judge_model.prepare_evaluation_data(
            queries, contexts, model_answers, gt_answers
        )
    )

    # 5. Save Final Results
    data_handler.save_final_output(evaluations, run_config, model_config, model_answers)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
