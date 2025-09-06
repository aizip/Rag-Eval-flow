from pathlib import Path
from visualizers.text_report import HTMLVisualizer
from visualizers.md_table_vis import MDVisualizer

# Register visualizers based on their scope
MODEL_VISUALIZERS = [
    HTMLVisualizer(),
]
GLOBAL_VISUALIZERS = [
    MDVisualizer(),
]

def main():
    results_root_dir = Path("../results/")
    
    model_dir = Path("/your/path/here")
    all_models_dir = Path("/your/path/here")
    output_dir = Path("./visualizations/")
    output_dir.mkdir(exist_ok=True)

    for vis in MODEL_VISUALIZERS:
        vis.generate(model_dir, output_dir)
    
    for vis in GLOBAL_VISUALIZERS:
        vis.generate(all_models_dir, output_dir)
    
    
    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()