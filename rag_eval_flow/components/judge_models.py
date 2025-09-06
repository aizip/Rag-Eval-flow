import os
import json
import time
from pathlib import Path # Added import
from openai import OpenAI
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from utils.prompts import format_judge_system_prompt, format_judge_user_prompt, METRIC_CONFIG

class BaseJudge(ABC):
    def __init__(self, metric: str, judge_model_name: str, **kwargs):
        self.metric = metric
        self.judge_model_name = judge_model_name
        if self.metric not in self.get_available_metrics():
            raise ValueError(f"Metric '{self.metric}' is not supported. Available: {self.get_available_metrics()}") # Corrected quoting
        self.metric_requirements = METRIC_CONFIG[self.metric]

    @abstractmethod
    def prepare_evaluation_data(self, original_queries: List[str], contexts: List[List[str] | List[Dict[str, str]] | str], 
                                model_answers: List[str], gt_answers: List[str]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def evaluate(self, prepared_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def get_available_metrics(self):
        return METRIC_CONFIG.keys()

class OpenAIChatJudge(BaseJudge):
    def __init__(self, judge_model_name: str, metric: str, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.3, max_tokens: int = 512, frequency_penalty: float = 0.5,
                 batch_api_endpoint: str = "/v1/chat/completions", completion_window: str = "24h",
                 base_results_dir: str = "../results/", # For storing batch files
                 model_name_for_paths: str = "unknown_model", # Used for unique batch filenames
                 **kwargs):
        super().__init__(metric=metric, judge_model_name=judge_model_name, **kwargs)
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable '{api_key_env}'.") # Corrected quoting
        self.openai_client = OpenAI(api_key=api_key)
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.batch_api_endpoint = batch_api_endpoint
        self.completion_window = completion_window

        # Construct paths for batch files based on results_dir and other unique identifiers
        base_results_path = Path(base_results_dir) 

        # Ensure model_name_for_paths is sanitized for use in filenames
        sanitized_model_name = model_name_for_paths.replace('/', '_').replace(' ', '_') # Corrected string replace
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        self.batch_requests_dir = base_results_path / "judge_batch_requests"  
        self.batch_results_dir = base_results_path / "judge_batch_results"   
        
        self.batch_requests_dir.mkdir(parents=True, exist_ok=True) 
        self.batch_results_dir.mkdir(parents=True, exist_ok=True)   

        # Unique filename for this batch operation
        metric_filename_part = self.metric.replace(' ', '_') # Corrected string replace
        base_filename = f"{sanitized_model_name}_{metric_filename_part}_{timestamp}"

        self.batch_request_filename = self.batch_requests_dir / f"{base_filename}_batch_request.jsonl"  
        self.batch_output_filename = self.batch_results_dir / f"{base_filename}_batch_output.jsonl"    
        self.batch_error_filename = self.batch_results_dir / f"{base_filename}_batch_error.jsonl"

        # attempt to disable HTTP status prints 
        # TODO fix
        import logging
        logging.getLogger("openai").setLevel(logging.WARNING)      
        
        print(f"OpenAIChatJudge initialized. Batch requests: {self.batch_request_filename}, Batch results: {self.batch_output_filename}") 

    def prepare_evaluation_data(self, original_queries: List[str], contexts: List[List[str] | List[Dict[str, str]] | str],
                                model_answers: List[str], gt_answers: List[str]) -> List[Dict[str, Any]]:
        eval_data_for_batch = []
        if not (len(original_queries) == len(contexts) == len(model_answers)):
            raise ValueError("Queries, contexts, and model_answers must have the same length.")

        for i in range(len(original_queries)):
            system_prompt = format_judge_system_prompt(self.metric, self.metric_requirements)
            data_dict = {
                "query" : original_queries[i],
                "context" : contexts[i],
                "ground_truth" : gt_answers[i]
            }
            user_prompt = format_judge_user_prompt(self.metric_requirements, model_answers[i], **data_dict)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # openai API specific construction
            eval_data_for_batch.append({
                "custom_id": f"sample_{i}_{self.metric.replace(' ', '_')}", # Corrected string replace
                "messages": messages
            })
        return eval_data_for_batch

    def _send_to_gpt_batch(self, tasks_to_send: List[Dict[str, Any]]) -> str | None:
        """Internal method to write batch file, send it, and monitor."""
        with open(self.batch_request_filename, "w") as f:
            for task_details in tasks_to_send: # task_details is {"custom_id": ..., "messages": ...}
                # Format each task for the OpenAI batch API
                formatted_task = self._format_batch_request_task(
                    custom_id=task_details["custom_id"],
                    messages=task_details["messages"],
                    model=self.judge_model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    frequency_penalty=self.frequency_penalty,
                    api_endpoint=self.batch_api_endpoint
                )
                f.write(json.dumps(formatted_task) + "\n") 
        
        print(f"Sending {len(tasks_to_send)} samples to OpenAI Batch API ({self.judge_model_name}) for metric '{self.metric}'. Request file: {self.batch_request_filename}") # Corrected quoting

        try:
            with open(self.batch_request_filename, "rb") as f_rb: 
                batch_input_file = self.openai_client.files.create(
                    file=f_rb,
                    purpose="batch"
                )
        except Exception as e:
            print(f"Error uploading batch request file {self.batch_request_filename}: {e}")
            raise

        try:
            batch_job = self.openai_client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint=self.batch_api_endpoint, # e.g. "/v1/chat/completions" for openai
                completion_window=self.completion_window 
            )
        except Exception as e:
            print(f"Error creating batch job with file ID {batch_input_file.id}: {e}")
            raise

        batch_id = batch_job.id
        
        print(f"Batch job created with ID: {batch_id}. Waiting for completion...")

        pbar = None
        try:
            # It can take a moment for OpenAI to initialize the job and determine the total request count.
            # We poll briefly until the total is available before creating the progress bar.
            while batch_job.request_counts.total == 0:
                time.sleep(5) 
                batch_job = self.openai_client.batches.retrieve(batch_id=batch_id)

            # init
            pbar = tqdm(total=batch_job.request_counts.total, desc=f"Batch {batch_id}", unit="req")

            # Main polling loop to update the progress bar.
            while batch_job.status not in ["completed", "failed", "cancelled", "expired"]:
                time.sleep(30)  
                batch_job = self.openai_client.batches.retrieve(batch_id=batch_id)

                completed_requests = batch_job.request_counts.completed
                failed_requests = batch_job.request_counts.failed
                current_progress = completed_requests + failed_requests

                pbar.n = current_progress
                pbar.set_postfix(
                    status=batch_job.status,
                    completed=completed_requests,
                    failed=failed_requests,
                    refresh=True  
                )
        finally:
            if pbar:
                # Perform a final update to make sure the bar shows the final numbers.
                final_completed = batch_job.request_counts.completed
                final_failed = batch_job.request_counts.failed
                pbar.n = final_completed + final_failed
                pbar.set_postfix(
                    status=batch_job.status,
                    completed=final_completed,
                    failed=final_failed,
                    refresh=True
                )
                pbar.close()
        
        if batch_job.status == "completed":
            print(f"Batch job {batch_id} completed.")
            return batch_job.output_file_id
        elif batch_job.status == "failed":
            print(f"Batch job {batch_id} failed. Error file ID: {batch_job.error_file_id}")
            # Optionally, retrieve and log errors from batch_job.errors
            if batch_job.error_file_id:
                return f"ERROR_FILE:{batch_job.error_file_id}" # Special marker
            raise Exception(f"OpenAI Batch job {batch_id} failed.")
        else: # Cancelled or other unexpected status
            raise Exception(f"OpenAI Batch job {batch_id} ended with status: {batch_job.status}.")
        
    def _format_batch_request_task(self,
        custom_id: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        frequency_penalty: float,
        api_endpoint: str = "/v1/chat/completions",
    ) -> Dict:
        """
        Format a single task with structured JSON outputs for the OpenAI Batch API.

        Args:
            custom_id: A unique identifier for this task (e.g., "sample_123").
            messages: The list of messages for the chat completion.
            model: The model to use (e.g., "gpt-4o").
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            frequency_penalty: Frequency penalty.
            api_endpoint (str): The API endpoint for the request.

        Returns:
            A dictionary representing the task for the OpenAI Batch API.
        """
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": api_endpoint,
            "body": {
                "model": model,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "explanation": {
                                    "type": "string",
                                    "description": "A detailed explanation of the evaluation result.",
                                },
                                "score": {
                                    "type": "number",
                                    "description": "The score retrieved from the evaluation.",
                                },
                            },
                            "required": ["explanation", "score"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                # Add other generation parameters if needed
            },
        }


    def _process_batch_results(self, output_file_id: Optional[str], error_file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        processed_evaluations = []
        target_output_file = None
        is_error_file = False

        if output_file_id:
            target_output_file = self.batch_output_filename 
            file_to_download = output_file_id
            print(f"Retrieving batch output content from file ID: {output_file_id}")
        elif error_file_id:
            target_output_file = self.batch_error_filename 
            file_to_download = error_file_id
            is_error_file = True
            print(f"Retrieving batch error content from file ID: {error_file_id}")
        else:
            print("No output or error file ID provided for batch processing.")
            return [] 

        try:
            file_content_response = self.openai_client.files.content(file_to_download)
            # The response might be a `openai.types.file_content.FileContent` object or similar
            # Access content via `.text` or `.content` depending on API version and response type
            if hasattr(file_content_response, 'text'):
                file_content = file_content_response.text
            elif hasattr(file_content_response, 'content'): # For binary content
                file_content = file_content_response.content.decode('utf-8') # TODO: don't assume utf-8
            else: # Fallback for older or different response structures
                file_content = str(file_content_response) 

            with open(target_output_file, "w") as f: 
                f.write(file_content)
            print(f"Batch {'error' if is_error_file else 'output'} saved to: {target_output_file}") 
        except Exception as e:
            print(f"Error retrieving or saving file content for file ID {file_to_download}: {e}")
            return [] 

        if is_error_file:
            print(f"Batch job resulted in an error file. See {target_output_file} for details.") 
            return [] 

        # Process the successful output file (JSONL format expected)
        with open(target_output_file, "r") as f: 
            for line in f:
                try:
                    result_item = json.loads(line.strip())
                    # structure of result_item from openai batch:
                    # {
                    #   "custom_id": "sample_123",
                    #   "response": {
                    #     "status_code": 200,
                    #     "request_id": "...",
                    #     "body": {
                    #       "id": "chatcmpl-...",
                    #       "object": "chat.completion",
                    #       "created": ...,
                    #       "model": "gpt-4o",
                    #       "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"explanation\": \"...\", \"score\": ...}"}, ...}]
                    #       ...
                    #     }
                    #   },
                    #   "error": null
                    # }
                    if result_item.get("response") and result_item["response"].get("body") and \
                       result_item["response"]["body"].get("choices") and \
                       len(result_item["response"]["body"]["choices"]) > 0:
                        
                        content_str = result_item["response"]["body"]["choices"][0]["message"]["content"]
                        # explanation: ... , score: ...
                        try:
                            evaluation_json = json.loads(content_str)
                            processed_evaluations.append({
                                "custom_id": result_item.get("custom_id"),
                                "explanation": evaluation_json.get("explanation"),
                                "score": evaluation_json.get("score"),
                                "raw_judge_response": content_str 
                            })
                        except json.JSONDecodeError as je:
                            print(f"Error decoding JSON content from judge for custom_id {result_item.get('custom_id')}: {je}. Content: '{content_str}'") # Corrected quoting
                            processed_evaluations.append({
                                "custom_id": result_item.get("custom_id"),
                                "error": "Failed to parse judge JSON output",
                                "raw_judge_response": content_str
                            })
                    elif result_item.get("error"):
                        print(f"Error in batch item for custom_id {result_item.get('custom_id')}: {result_item.get('error')}")
                        processed_evaluations.append({
                            "custom_id": result_item.get("custom_id"),
                            "error": result_item.get("error")
                        })
                    else:
                        print(f"Unexpected structure in batch result item for custom_id {result_item.get('custom_id')}")
                        processed_evaluations.append({
                            "custom_id": result_item.get("custom_id"),
                            "error": "Malformed batch result item"
                        })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line from batch output file {target_output_file}: {e}. Line: '{line.strip()}'") # Corrected quoting, Path object stringifies
                    # TODO: better handle malformed lines
                    processed_evaluations.append({
                        "custom_id": f"unknown_malformed_line_{len(processed_evaluations)}",
                        "error": "Malformed line in batch output file",
                        "raw_line": line.strip()
                    })
        
        return processed_evaluations


    def evaluate(self, prepared_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sends prepared data to OpenAI Batch API and processes results.
        prepared_data is a list of dicts, each with "custom_id" and "messages".
        """
        if not prepared_data:
            print("No data provided to evaluate.")
            return []

        try:
            output_file_id_or_error_marker = self._send_to_gpt_batch(prepared_data)
            
            output_file_id = None
            error_file_id = None

            if output_file_id_or_error_marker.startswith("ERROR_FILE:"):
                error_file_id = output_file_id_or_error_marker.split(":",1)[1]
            else:
                output_file_id = output_file_id_or_error_marker
                
            evaluations = self._process_batch_results(output_file_id, error_file_id)
            
            # remember to do order checknig
            
            # for later lookup
            eval_map = {eval_item.get("custom_id"): eval_item for eval_item in evaluations}
            
            # reorder
            ordered_results = []
            for item in prepared_data:
                custom_id = item.get("custom_id")
                if custom_id in eval_map:
                    ordered_results.append(eval_map[custom_id])
                else:
                    # should ideally not happen if all items were processed
                    ordered_results.append({
                        "custom_id": custom_id,
                        "error": "Evaluation result not found for this item after batch processing."
                    })
            return ordered_results

        except Exception as e:
            print(f"An error occurred during the OpenAI batch evaluation process: {e}")
            return [{ "custom_id": item.get("custom_id"), "error": str(e) } for item in prepared_data]

# TODO: add more judge api support (vercel/gemini, claude, deepseek, etc.)
class DeepSeekChatJudge(BaseJudge):
    def __init__(self, judge_model_name: str, metric: str, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.3, max_tokens: int = 512, frequency_penalty: float = 0.5,
                 batch_api_endpoint: str = "/v1/chat/completions", completion_window: str = "24h",
                 base_results_dir: str = "../results/judge_batches", # For storing batch files
                 model_name_for_paths: str = "unknown_model", # Used for unique batch filenames
                 **kwargs):
        super().__init__(metric=metric, judge_model_name=judge_model_name, **kwargs)
        raise NotImplementedError
    
    def prepare_evaluation_data(self, original_queries: List[str], contexts: List[List[str] | List[Dict[str, str]] | str], 
                                model_answers: List[str], gt_answers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def evaluate(self, prepared_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError


# example:
# judge = OpenAIChatJudge(judge_model_name="gpt-4o", metric="response_coherence", base_results_dir="./temp_judge_batches")
# data_to_eval = judge.prepare_evaluation_data(
#     original_queries=["Q1"], 
#     contexts=["C1"], 
#     model_answers=["A1"],
# )
# results = judge.evaluate(data_to_eval)
# print(results)
