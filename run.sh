#!/bin/bash

# Default values
DATA_PATH="YOUR_DATA_PATH"
MODEL_NAME="google/gemma-3-1b-it"
MODEL_TYPE="causal_lm"
INPUT_COLUMN="question" 
DOCUMENT_COLUMN="contexts"
ANSWER_COLUMN="answer"
SAMPLE_SIZE=1000
BATCH_SIZE=16
MAX_NEW_TOKENS=512
OUTPUT_PATH_DIR="./batch_results/"
METRIC="query relevance"
GPT_MODEL="gpt-4o"
SEED=42
API_KEY=""
RECOVERY_MODE=false
BATCH_RESULTS_PATH=""

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --data_path PATH       Path to data file (JSON, CSV, Parquet)"
    echo "  --model_name NAME      HuggingFace model name"
    echo "  --model_type TYPE      Model type: 'causal_lm' or 'seq2seq' (default: causal_lm)"
    echo "  --input_column COL     Column name with input queries"
    echo "  --document_column COL  Column name with input documents"
    echo "  --gt_column COL        Key name in dataset containing ground truth answers"
    echo "  --sample_size SIZE     Number of samples to evaluate (default: 1000)"
    echo "  --batch_size SIZE      Batch size for transformer processing (default: 16)"
    echo "  --max_new_tokens NUM   Maximum new tokens to generate (default: 512)"
    echo "  --output_path_dir DIR  Path to save results (default: ./batch_results/)"
    echo "  --metric METRIC        Single metric to evaluate (e.g., 'response coherence')"
    echo "  --gpt_model MODEL      GPT model to use (default: gpt-4o)"
    echo "  --seed NUM             Random seed (default: 42)"
    echo "  --api_key KEY          OpenAI API key (optional, can use env var)"
    echo "  --recovery_mode        Enable recovery mode to process existing batch results"
    echo "  --batch_results_path P Path to existing batch results file (used with recovery_mode)"
    echo "  -h, --help             Display this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --input_column)
            INPUT_COLUMN="$2"
            shift 2
            ;;
        --document_column)
            DOCUMENT_COLUMN="$2"
            shift 2
            ;;
        --gt_column)
            ANSWER_COLUMN="$2"
            shift 2
            ;;
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --output_path_dir)
            OUTPUT_PATH_DIR="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --gpt_model)
            GPT_MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --recovery_mode)
            RECOVERY_MODE=true
            shift
            ;;
        --batch_results_path)
            BATCH_RESULTS_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Build the command
CMD="python generic_model_eval.py \
    --data_path \"$DATA_PATH\" \
    --model_name \"$MODEL_NAME\" \
    --model_type \"$MODEL_TYPE\" \
    --input_column \"$INPUT_COLUMN\" \
    --document_column \"$DOCUMENT_COLUMN\" \
    --sample_size \"$SAMPLE_SIZE\" \
    --batch_size \"$BATCH_SIZE\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --output_path_dir \"$OUTPUT_PATH_DIR\" \
    --gpt_model \"$GPT_MODEL\" \
    --seed \"$SEED\""

# Add optional arguments if they're set
if [ ! -z "$METRIC" ]; then
    CMD="$CMD --metric \"$METRIC\""
fi

if [ ! -z "$ANSWER_COLUMN" ]; then
    CMD="$CMD --gt_column \"$ANSWER_COLUMN\""
fi

if [ ! -z "$API_KEY" ]; then
    CMD="$CMD --api_key \"$API_KEY\""
fi

if [ "$RECOVERY_MODE" = true ]; then
    CMD="$CMD --recovery_mode"
    
    if [ ! -z "$BATCH_RESULTS_PATH" ]; then
        CMD="$CMD --batch_results_path \"$BATCH_RESULTS_PATH\""
    fi
fi

# Run the Python script with the specified parameters
echo "Running evaluation with the following parameters:"
echo "Data Path: $DATA_PATH"
echo "Model Name: $MODEL_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Input Column: $INPUT_COLUMN"
echo "Document Column: $DOCUMENT_COLUMN"
echo "Ground Truth Column: $ANSWER_COLUMN"
echo "Sample Size: $SAMPLE_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Output Path: $OUTPUT_PATH_DIR"
echo "Metric: $METRIC"
echo "GPT Model: $GPT_MODEL"
echo "Seed: $SEED"
if [ "$RECOVERY_MODE" = true ]; then
    echo "Recovery Mode: Enabled"
    echo "Batch Results Path: $BATCH_RESULTS_PATH"
fi
echo "========================="
echo ""

# Execute the command
eval $CMD