from pathlib import Path
from visualizers.text_report import HTMLVisualizer
# from visualizers.markdown_visualizer import MarkdownVisualizer

# Register visualizers based on their scope
MODEL_VISUALIZERS = [
    HTMLVisualizer(),
]
# GLOBAL_VISUALIZERS = [
#     MarkdownVisualizer(),
# ]

def main():
    results_root_dir = Path("../results/")
    
    model_dir = Path("/home/oliver/Rag-Eval-flow/results/models/autobacs/gemini2.5-flash")
    # model_dir = Path("./results/models/autobacs/deepseekv3")
    output_dir = Path("../visualizations/")
    output_dir.mkdir(exist_ok=True)

    for vis in MODEL_VISUALIZERS:
        vis.generate(model_dir, output_dir)

    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()