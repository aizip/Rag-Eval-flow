# Rag-Eval-flow 

A flexible, configuration-driven framework for running evaluations on language models.

> **🚧 Active Development on `development` Branch 🚧**
>
> This branch contains a significant refactoring of the framework. It is unstable and subject to breaking changes. 
-----

## Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/aizip/Rag-Eval-flow.git
    cd Rag-Eval-flow
    ```

2.  **Install dependencies:**
    Install the base package along with the optional backends you need.

    ```sh
    # Install with Hugging Face and OpenAI support
    pip install -e .[huggingface,openai]

    # Install with llama-cpp-python support
    pip install -e .[llama-cpp]
    ```

-----

## Quickstart

The easiest way to run an evaluation is with the provided script, which uses the `default_rag_evaluation_test` pipeline from the config file.

```sh
bash run.sh
```

You can customize the run by editing the variables in `config/evaluation_config.yaml` (preferred) or by using the `rag_eval_flow/main.py` arguments directly.

## How to Customize the Configuration

All evaluation logic is controlled by `config/evaluation_config.yaml`. You can add your own data, models, and evaluation pipelines directly to this file.

#### 1\. Define Your Model

Add a new entry under the `models` section. You can copy an existing configuration like `huggingface_causal_lm` and change the details.

#### 2\. Define Your Judge

Similarly, you can define a new judge or a new configuration for an existing one under the `judges` section.

#### 3\. Create a New Evaluation Pipeline

This is where you tie everything together. Add a new entry under `evaluation_pipelines`. This new pipeline key is what you will pass to the run script.

**Example:** Adding a new pipeline to test `Mistral-7B` on a custom metric.

```yaml
# In config/evaluation_config.yaml

# ... existing models and judges sections ...

evaluation_pipelines:
  # Add your new pipeline here
  mistral_custom_eval:
    data_source_key: "defaultJSONL" # Which data format to use
    data_path: "/path/to/your/data.jsonl" # Path to your data
    
    model_key: "huggingface_causal_lm" # Which model backend to use
    model_config_overrides:
      model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.2" # The specific model make sure you have any API keys you need in your environment!

    judge_key: "openai_chat_judge" # Which judge to use
    judge_config_overrides:
      judge_model_name: "gpt-4o" 
    
    metric: "your_custom_metric_name" # The metric to be evaluated
    sample_size: 200 # Number of samples to run
```


To run the custom pipeline you just created, edit the `PIPELINE_KEY` in `run.sh` and execute it.

```sh
# In run.sh
PIPELINE_KEY="mistral_custom_eval"
```

```sh
# Then run from your terminal
bash run.sh
```