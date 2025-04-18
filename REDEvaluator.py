import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os
import json
import argparse
import time
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from utils.prompts import format_batch_request_task, format_messages, format_rag_prompt


class REDEvaluator:
    def __init__(
        self,
        model_name: str,
        input_column: str,
        document_column: str,
        model_type: str = "causal_lm",  # Options: "causal_lm" for now!
        sample_size: int = 1000,
        random_seed: int = 42,
        batch_size: int = 16,
        device: str = None,
        openai_api_key: str = None,
        gpt_model: str = "gpt-4o",
        max_new_tokens: int = 512,
        metric: str = None,
    ):
        """
        Initialize the evaluator with model and processing parameters.

        Args:
            model_name: The HuggingFace model identifier to use
            input_column: Column name in the DataFrame containing text to process
            model_type: Type of model to use ("causal_lm" for now!)
            sample_size: Number of samples to process (default: 1000)
            random_seed: Random seed for reproducibility (default: 42)
            batch_size: Batch size for transformer model inference (default: 16)
            openai_api_key: OpenAI API key (default: None, will use env var)
            gpt_model: GPT model to use for evaluation (default: "gpt-4o")
            max_new_tokens: Maximum number of new tokens to generate (default: 128)
        """
        self.model_name = model_name
        self.input_column = input_column
        self.document_column = document_column
        self.model_type = model_type
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_requests_dir = "./batch_requests/"
        self.batch_results_dir = "./batch_results/"
        self.eval_results_dir = "./eval_results/"
        self.single_metric = metric

        # Make all the necessary directories if they don't exist
        os.makedirs(self.batch_requests_dir, exist_ok=True)
        os.makedirs(self.batch_results_dir, exist_ok=True)
        os.makedirs(self.eval_results_dir, exist_ok=True)
        # Set up OpenAI client
        if not openai_api_key:
            open_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(
            api_key=open_api_key
        )
        self.gpt_model = gpt_model

        self.batch_output_fn = os.path.join(
            self.batch_results_dir, os.path.basename(self.model_name) + ".jsonl"
        )
        self.final_output_fn = os.path.join(
            self.eval_results_dir, os.path.basename(self.model_name) + ".jsonl"
        )

    def sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample rows from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Sampled DataFrame
        """
        np.random.seed(self.random_seed)  # eval consistency

        if len(df) <= self.sample_size:
            print(
                f"DataFrame has {len(df)} rows, which is less than the requested sample size of {self.sample_size}."
            )
            print("Using the entire DataFrame.")
            return df

        sampled_df = df.sample(self.sample_size, random_state=self.random_seed)
        print(f"Sampled {len(sampled_df)} rows from {len(df)} total rows.")
        return sampled_df

    def generate_answers(self, query_texts: List[str], documents) -> List[str]:
        """
        Generate text answers using the transformer model for a list of queries.

        Args:
            query_texts: List of input queries

        Returns:
            List of generated text answers
        """
        assert len(query_texts) == len(documents)
        accepts_sys = "System role not supported" not in self.tokenizer.chat_template
        generated_texts = []

        # Process in batches
        for i in tqdm(
            range(0, len(query_texts), self.batch_size), desc="Generating answers"
        ):
            batch_queries = query_texts[i : i + self.batch_size]
            batch_contexts = documents[i : i + self.batch_size]
            combined = [
                format_rag_prompt(query, context, accepts_sys)
                for query, context in zip(batch_queries, batch_contexts)
            ]

            # Format the input for the model
            inputs = self.tokenizer.apply_chat_template(
                combined,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                padding="longest",
                truncation=True,
                max_length=2048,
                add_generation_prompt=True,
            ).to(self.device)

            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Use greedy decoding for deterministic outputs
                    num_return_sequences=1,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Decode outputs
            if self.model_type == "causal_lm":
                # For causal models, we need to remove the input tokens
                input_length = inputs.input_ids.shape[1]
                decoded_outputs = [
                    self.tokenizer.decode(
                        output[input_length:], skip_special_tokens=True
                    )
                    for output in outputs
                ]
            else:
                # For seq2seq models, just decode the entire output
                decoded_outputs = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]

            generated_texts.extend(decoded_outputs)

        return generated_texts

    def prepare_gpt_evaluation_data(
        self, original_queries: List[str], contexts: List[str], model_answers: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Prepare data for GPT-4 evaluation of answer correctness.

        Args:
            original_queries: Original input queries
            model_answers: Generated answers from the model

        Returns:
            List of messages for GPT-4 batch evaluation
        """
        eval_data = []

        for i, (query, context, answer) in enumerate(
            zip(original_queries, contexts, model_answers)
        ):
            # Create evaluation prompt
            messages = format_messages(
                query=query,
                context=context,
                response=answer,
                metric_name=self.single_metric,
            )

            eval_data.append({"id": f"sample_{i}", "messages": messages})

        return eval_data

    def send_to_gpt4_batch(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send data to GPT-4 using the batch API.

        Args:
            evaluation_data: List of evaluation messages

        Returns:
            GPT-4 batch responses
        """
        tasks = []

        generation_args = {
            "model": self.gpt_model,
            "temperature": 0.3,
            "max_tokens": 512,
            "frequency_penalty": 0.5,
        }
        for datapoint in evaluation_data:
            task = format_batch_request_task(
                metric_name=self.single_metric,
                eval_datapoint=datapoint,
                generation_args=generation_args,
            )
            tasks.append(task)

        fn = os.path.join(
            self.batch_requests_dir, os.path.basename(self.model_name) + ".jsonl"
        )
        with open(fn, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

        print(f"Sending {len(evaluation_data)} samples to GPT-4 for evaluation...")

        batch_file = self.openai_client.files.create(
            file=open(fn, "rb"), purpose="batch"
        )

        batch_response = self.openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Wait for completion
        batch_id = batch_response.id
        print(f"Batch job created with ID: {batch_id}")

        # Monitor progress
        status = ""
        while status != "completed":
            time.sleep(10)  # Check every 10 seconds
            job = self.openai_client.batches.retrieve(batch_id=batch_id)
            completed = job.status
            stats = job.request_counts
            if completed != status:
                print(f"Batch job status: {completed}")
                print("Counts: ", stats)
                status = completed

        # Retrieve results
        results = self.openai_client.batches.retrieve(batch_id=batch_id)
        return results

    def process_existing_batch_results(self, batch_results_path: str) -> Dict[str, Any]:
        """
        Process existing batch results without re-running the evaluation pipeline.

        Args:
            batch_results_path: Path to the existing batch results file

        Returns:
            Dictionary with processed results
        """

        print(f"Loading batch results from {batch_results_path}...")

        results = []
        evals = []

        # Load and parse the batch results file
        with open(batch_results_path, "r") as file:
            for line in file:
                try:
                    json_object = json.loads(line.strip())
                    results.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    continue

        print(f"Loaded {len(results)} results from batch file")

        # Process each result
        for result in results:
            try:
                gpt_output = json.loads(
                    result["response"]["body"]["choices"][0]["message"]["content"]
                )
                evals.append(gpt_output)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing result: {e}")
                continue

        return {"evaluations": evals}

    def process_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main processing and evaluation workflow with caching support.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with results
        """
        # Sample data deterministically
        sampled_df = self.sample_dataframe(df)

        # Create a cache filename based on model, seed, and sample size
        cache_dir = os.path.join(self.batch_requests_dir, "response_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = os.path.join(
            cache_dir,
            f"{os.path.basename(self.model_name)}_seed{self.random_seed}_n{self.sample_size}.json",
        )

        # Check if cache exists
        if os.path.exists(cache_filename):
            print(f"Loading cached responses from {cache_filename}")

            cache_df = pd.read_json(cache_filename)

            queries = cache_df["queries"].tolist()
            contexts = cache_df["contexts"].tolist()
            model_answers = cache_df["model_answers"].tolist()
            # Load the original dataframe indexes from cache
            # df_indexes = cache_df['df_indexes'].tolist()

            print(f"Loaded {len(model_answers)} cached responses")
        else:
            # Extract input queries and contexts
            queries = sampled_df[self.input_column].tolist()
            contexts = sampled_df[self.document_column].tolist()
            # Save the original indexes for later reference
            df_indexes = sampled_df.index.tolist()
            print(f"Using device: {self.device}")
            print(f"Loading model: {self.model_name}")
            print(f"Model type: {self.model_type}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load appropriate model type
            if self.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    #attn_implementation="flash_attention_2",  # Change if you have it
                    device_map=self.device,
                )
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            elif self.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
                    self.device
                )
            else:
                raise ValueError(
                    f"Unsupported model type: {self.model_type}. Use 'causal_lm' or 'seq2seq'."
                )

            # Generate
            print(
                f"Generating answers for {len(queries)} queries using {self.model_name}..."
            )
            model_answers = self.generate_answers(queries, contexts)

            # Create cached DF
            cache_df = pd.DataFrame(
                {
                    "queries": queries,
                    "contexts": contexts,
                    "model_answers": model_answers,
                    "df_indexes": df_indexes,
                    "model_name": [self.model_name] * len(queries),
                    "timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")] * len(queries),
                }
            )

            print(f"Saving responses to cache: {cache_filename}")
            cache_df.to_json(cache_filename, orient="records")

        # Create combined results for reference using the preserved indexes
        response_data = []
        for query, context, answer in zip(queries, contexts, model_answers):
            response_data.append(
                {
                    # "index": idx,  # Use the preserved index from the sampled dataframe
                    "query": query,
                    "context": context,
                    "model_answer": answer,
                }
            )

        # Save response data to file for recovery
        response_fn = os.path.join(
            self.batch_results_dir,
            os.path.basename(self.model_name) + "_responses.jsonl",
        )
        temp_response_df = pd.DataFrame(response_data)
        temp_response_df.to_json(response_fn, orient="records", lines=True)
        print(f"Response data saved to {response_fn}")

        # Prepare evaluation data
        eval_data = self.prepare_gpt_evaluation_data(queries, contexts, model_answers)

        # Send to GPT-4
        evaluation_results = self.send_to_gpt4_batch(eval_data)
        results = []
        evals = []

        # Parse GPT-4 responses
        try:  # Check for batch errors
            result = self.openai_client.files.content(
                evaluation_results.output_file_id
            ).content
            with open(self.batch_output_fn, "wb") as file:
                file.write(result)
        except Exception as e:
            result = self.openai_client.files.content(
                evaluation_results.error_file_id
            ).content
            with open(self.batch_output_fn[:-6] + "_error.jsonl", "wb") as file:
                file.write(result)
            file.close()
            return evals

        # Return combined results
        with open(self.batch_output_fn, "r") as file:
            for line in file:
                json_object = json.loads(line.strip())
                results.append(json_object)

        for result in results:
            gpt_output = json.loads(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )
            evals.append(gpt_output)

        return {"response_data": response_data, "evaluations": evals}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model answers with GPT-4"
    )
    parser.add_argument(
        "--metric", help="Single metric to evaluate (e.g., 'response coherence')"
    )
    parser.add_argument(
        "--data_path", required=True, help="Path to the JSON, CSV, or Parquet data file"
    )
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument(
        "--model_type",
        default="causal_lm",
        choices=["causal_lm", "seq2seq"],
        help="Model type: 'causal_lm' for generative models, 'seq2seq' for encoder-decoder models",
    )
    parser.add_argument(
        "--input_column", required=True, help="Column name with input queries"
    )
    parser.add_argument(
        "--document_column", required=True, help="Column name with input queries"
    )
    parser.add_argument(
        "--sample_size", type=int, default=1000, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output_path_dir", default="./batch_results/", help="Path to save results"
    )
    parser.add_argument("--api_key", help="OpenAI API key (optional, can use env var)")
    parser.add_argument("--gpt_model", default="gpt-4o", help="GPT model to use")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for transformer processing",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--recovery_mode",
        action="store_true",
        help="Enable recovery mode to process existing batch results",
    )
    parser.add_argument(
        "--batch_results_path",
        help="Path to existing batch results file (used with recovery_mode)",
    )

    args = parser.parse_args()

    evaluator = REDEvaluator(
        model_name=args.model_name,
        input_column=args.input_column,
        document_column=args.document_column,
        model_type=args.model_type,
        sample_size=args.sample_size,
        random_seed=args.seed,
        batch_size=args.batch_size,
        openai_api_key=args.api_key,
        gpt_model=args.gpt_model,
        max_new_tokens=args.max_new_tokens,
        metric=args.metric,
    )

    # Process in recovery mode or normal mode
    if args.recovery_mode:
        if not args.batch_results_path:
            raise ValueError("batch_results_path is required when using recovery mode")

        print("Running in recovery mode...")
        results = evaluator.process_existing_batch_results(args.batch_results_path)
    else:
        # Load data
        print(f"Loading data from {args.data_path}")
        if args.data_path.endswith(".csv"):
            df = pd.read_csv(args.data_path)
        elif args.data_path.endswith(".parquet"):
            df = pd.read_parquet(args.data_path)
        elif args.data_path.endswith(".json"):
            df = pd.read_json(args.data_path)
        elif args.data_path.endswith(".jsonl"):
            df = pd.read_json(args.data_path, lines=True)
        else:
            raise ValueError("Data file must be CSV, Parquet, or JSON format")

        print(
            f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()}"
        )

        # Process and evaluate
        results = evaluator.process_and_evaluate(df)

    # Save results
    if args.recovery_mode:
        outputfn = evaluator.final_output_fn.replace(".jsonl", "_recovery.jsonl")
    elif evaluator.single_metric:
        # Create model-specific directory under metrics_testing
        model_dir = os.path.join(
            evaluator.eval_results_dir, os.path.basename(evaluator.model_name)
        )

        # Create the directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")

        # Create the output filename with metric name
        outputfn = os.path.join(model_dir, f"{evaluator.single_metric}.jsonl")
    else:
        outputfn = evaluator.final_output_fn

    # Ensure parent directory exists (in case directories above weren't created yet)
    os.makedirs(os.path.dirname(outputfn), exist_ok=True)

    # Save results
    temp_df = pd.DataFrame(results)
    temp_df.to_json(outputfn, orient="records", lines=True)

    print(f"Evaluation complete. Results saved to {outputfn}")


if __name__ == "__main__":
    main()
