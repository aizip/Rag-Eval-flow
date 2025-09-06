from abc import ABC, abstractmethod
from .interface import VisualizerInterface
from pathlib import Path
import pandas as pd
from rag_eval_flow.utils.data_utils import construct_one_model_metrics_df, find_latest_metric_files, compute_rejection_rates, assemble_accurate_df

class MDVisualizer(VisualizerInterface):
    @property
    def description(self) -> str:
        """A one-sentence description of what the visualizer produces."""
        return "Generates a Markdown Table of average scores for all models under the given directory."


    def generate(self, input_path: Path, output_dir: Path) -> Path:
        """
        Generates a Markdown table comparing the average scores of different models.

        This function iterates through model-specific subdirectories within the `input_path`,
        collates evaluation data using the `construct_one_model_metrics_df` function,
        calculates the average score for each discovered metric, and then compiles all
        results into a single, easy-to-read Markdown summary table.

        Args:
            input_path: A Path object pointing to the root directory that contains
                        subdirectories for each model's evaluation files.
            output_dir: A Path object for the directory where the generated
                        Markdown report will be saved.

        Returns:
            A Path object that points to the newly created Markdown file.
        """
        all_model_results = []
        all_metric_names = set()

        # Assume each subdirectory in the input path corresponds to a unique model.
        # Sort the directories to ensure the output is consistently ordered.
        model_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

        for model_dir in model_dirs:
            model_name = model_dir.name
            # Locate all evaluation files, assuming a common .jsonl format.
            evaluation_files = find_latest_metric_files(model_dir)

            if not evaluation_files:
                continue

            total_df, discovered_metrics = construct_one_model_metrics_df(evaluation_files)
            print(total_df.columns)
            try: # Use "refusal_presence" to calculate precision and negative accuracy
                tpr, tnr = compute_rejection_rates(total_df, return_df=False)
                tpr_title = "Positive-refusal Detection Accuracy"
                tnr_title = "Negative-refusal Detection Accuracy"
                merged_df = assemble_accurate_df(total_df)
            except ValueError:
                continue
            if merged_df.empty:
                continue

            all_metric_names.update(discovered_metrics)

            # Calculate and store the average score
            current_model_scores = {"Model": model_name}
            for metric in discovered_metrics:
                score_col = f"{metric}_score"
                if score_col in merged_df.columns:
                    # Convert the column to a numeric type, coercing errors into NaN.
                    # The .mean() function will automatically ignore these NaN values.
                    scores = pd.to_numeric(merged_df[score_col], errors="coerce")
                    avg_score = scores.mean()
                    # 3 SF max
                    current_model_scores[metric] = f"{avg_score:.2f}"
            
            current_model_scores.update({
                "Positive-refusal Detection Accuracy" : f"{tpr:.2f}",
                "Negative-refusal Detection Accuracy" : f"{tnr:.2f}"
            })

            all_metric_names.update([tpr_title, tnr_title])
            
            all_model_results.append(current_model_scores)

        # --- Markdown Table Generation ---

        # Ensure the output directory exists.
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / "metrics_summary.md"

        # If no data was processed, write a simple message and return.
        if not all_model_results:
            output_file_path.write_text("No model data was found to generate a summary report.")
            return output_file_path

        if "refusal_presence" in all_metric_names:
            all_metric_names.remove("refusal_presence")
        sorted_metrics = sorted(list(all_metric_names))
        
        # Create human-readable titles for the table header (e.g., 'query_relevance' -> 'Query Relevance').
        header_titles = [metric.replace("_", " ").title() for metric in sorted_metrics]
        
        # Use simple MD format, TODO: format name?
        header = f"| Model | {' | '.join(header_titles)} |"
        separator = f"|:---{'|:---' * len(sorted_metrics)}|"
        
        # Construct each data row for the table.
        table_rows = []
        sorted_results = sorted(all_model_results, key=lambda x: x['Model'])

        # Construct the table, "--" replaces NaN
        for model_data in sorted_results:
            row_values = [model_data['Model']]
            for metric in sorted_metrics:
                score = model_data.get(metric, "--")
                row_values.append(score)
            table_rows.append(f"| {' | '.join(row_values)} |")

        markdown_content = "\n".join([header, separator] + table_rows)
        output_file_path.write_text(markdown_content)

        return output_file_path