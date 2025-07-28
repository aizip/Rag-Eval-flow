# Rag-Eval-flow 

A flexible, configuration-driven framework for running evaluations on language models.

> **🚧 Active Development on `refactor` Branch 🚧**
>
> This branch contains a significant refactoring of the framework. It is unstable and subject to breaking changes. For a stable version, please use the `development` branch.

-----

## Installation

1.  **Clone the repository:**

    ```sh
    git clone <your-repository-url>
    cd Rag-Eval-flow
    ```

2.  **Install dependencies:**
    Install the base package along with the optional backends you need.

    ```sh
    # Install with Hugging Face and OpenAI support
    pip install .[huggingface,openai]

    # Install with Llama.cpp support
    pip install .[llama-cpp]
    ```

-----

## Quickstart

The easiest way to run an evaluation is with the provided script, which uses the `default_rag_evaluation_test` pipeline from the config file.

```sh
bash run.sh
```

You can customize the run by editing the variables in `config/evaluation_config.yaml` (preferred) or by using the `src/main.py` arguments directly.