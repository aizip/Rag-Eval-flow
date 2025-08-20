#!/bin/bash

echo "🚀 Starting evaluation process..."

# --- CONFIGURATION ---
# Paths are relative to this script's location (the project root)

CONFIG_PATH="config/vlm_evaluation_config.yaml" # please override inside the yaml, not here.

# Set the evaluation pipeline key you want to run from the config file.
PIPELINE_KEY="default_rag_evaluation_test"

CMD="python rag_eval_flow/main.py --config_path $CONFIG_PATH --pipeline_key $PIPELINE_KEY"

echo "----------------------------------------"
echo "Executing Command:"
echo "$CMD"
echo "----------------------------------------"

eval $CMD
