from pathlib import Path
from visualizers.text_report import HTMLVisualizer
<<<<<<< HEAD
# from visualizers.markdown_visualizer import MarkdownVisualizer
=======
from visualizers.md_table_vis import MDVisualizer
>>>>>>> 73427d9 ([feat] Implement visualization scripts and needed helpers v0.1)

# Register visualizers based on their scope
MODEL_VISUALIZERS = [
    HTMLVisualizer(),
]
<<<<<<< HEAD
# GLOBAL_VISUALIZERS = [
#     MarkdownVisualizer(),
# ]
=======
GLOBAL_VISUALIZERS = [
    MDVisualizer(),
]
>>>>>>> 73427d9 ([feat] Implement visualization scripts and needed helpers v0.1)

def main():
    results_root_dir = Path("../results/")
    
<<<<<<< HEAD
    model_dir = Path("/home/oliver/Rag-Eval-flow/results/models/autobacs/gemini2.5-flash")
    # model_dir = Path("./results/models/autobacs/deepseekv3")
    output_dir = Path("../visualizations/")
    output_dir.mkdir(exist_ok=True)

    for vis in MODEL_VISUALIZERS:
        vis.generate(model_dir, output_dir)
=======
    model_dir = Path("/home/oliver/Rag-Eval-flow/results/models/autobacs/gemma-2-2b-it_adapter_gmlora_lr5e5_b128_e2_jp20kfact_filtered_ABfact_concise")
    all_models_dir = Path("/home/oliver/Rag-Eval-flow/results/models/autobacs/")
    # model_dir = Path("./results/models/autobacs/deepseekv3")
    output_dir = Path("./visualizations/")
    output_dir.mkdir(exist_ok=True)

    # for vis in MODEL_VISUALIZERS:
    #     vis.generate(model_dir, output_dir)
    
    for vis in GLOBAL_VISUALIZERS:
        vis.generate(all_models_dir, output_dir)
    
    
>>>>>>> 73427d9 ([feat] Implement visualization scripts and needed helpers v0.1)

    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()