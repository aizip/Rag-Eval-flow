from abc import ABC, abstractmethod
from pathlib import Path

class VisualizerInterface(ABC):
    @property
    @abstractmethod
    def description(self) -> str:
        """A one-sentence description of what the visualizer produces."""
        pass

    @abstractmethod
    def generate(self, input_path: Path, output_dir: Path) -> Path:
        """
        Generates a visualization from a given input path.
        """
        pass