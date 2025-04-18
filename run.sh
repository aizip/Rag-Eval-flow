#!/bin/bash

# Default values
DATA_PATH="./data/RED6k-toy.jsonl"  # Replace with your actual data path
MODEL_NAME="your model here" # Replace with your actual model name
INPUT_COLUMN="question" 
DOCUMENT_COLUMN="contexts"
SAMPLE_SIZE=1000
BATCH_SIZE=16
METRIC="query relevance" # Replace with your actual metric

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --data_path PATH     Path to data JSON file"
    echo "  --model_name NAME    Name of the model to use"
    echo "  --input_column COL   Input column name"
    echo "  --document_column COL Column name for documents"
    echo "  --sample_size SIZE   Sample size to process"
    echo "  --batch_size SIZE    Batch size for processing"
    echo "  --metric METRIC      Metric to evaluate"
    echo "  -h, --help           Display this help message"
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
        --input_column)
            INPUT_COLUMN="$2"
            shift 2
            ;;
        --document_column)
            DOCUMENT_COLUMN="$2"
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
        --metric)
            METRIC="$2"
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

# Run the Python script with the specified parameters
echo "Running evaluation with the following parameters:"
echo "Data Path: $DATA_PATH"
echo "Model Name: $MODEL_NAME"
echo "Input Column: $INPUT_COLUMN"
echo "Document Column: $DOCUMENT_COLUMN"
echo "Sample Size: $SAMPLE_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Metric: $METRIC"
echo "========================="
echo ""

python REDEvaluator.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --input_column "$INPUT_COLUMN" \
    --document_column "$DOCUMENT_COLUMN" \
    --sample_size "$SAMPLE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --metric "$METRIC"