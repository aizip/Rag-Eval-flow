from utils.config_utils import get_config_value
import importlib

class ObjectFactory:
    def __init__(self, config):
        """
        Initializes the factory with the base configuration.
        Args:
            config (dict): The loaded YAML configuration.
        """
        self.config = config
        self.base_config = config.get("base_config", {})

    def create(self, object_key, config_section_key, **cli_overrides):
        """
        Creates an object (data_handler, model, judge) based on the configuration.

        Args:
            object_key (str): The specific key for the object within its section (e.g., "default_jsonl", "huggingface_causal_lm").
            config_section_key (str): The top-level key in the YAML config for this type of object (e.g., "data_sources", "models", "judges").
            **cli_overrides: Keyword arguments passed from the CLI or main script to override config values.

        Returns:
            An instance of the requested class, or None if creation fails.
        """
        # retrieve config tree
        object_configs = get_config_value(self.config, config_section_key)
        if not object_configs or object_key not in object_configs:
            print(f"Error: Configuration for '{object_key}' not found in section '{config_section_key}'.")
            return None
        specific_config = object_configs[object_key].copy() 

        # class instantiation
        class_name = specific_config.pop("class", None)
        if not class_name:
            print(f"Error: 'class' not specified for '{object_key}' in '{config_section_key}'.")
            return None

        # init args
        init_args = self.base_config.copy()
        init_args.update(specific_config) # Apply specific config from YAML
        init_args.update(cli_overrides)   # Apply runtime overrides

        # Filter out any keys that are not meant for the constructor if necessary,
        # or ensure constructors can handle **kwargs.

        try:
            module_name_part = config_section_key # key to .py conversion, TODO: handle more elegantly?
            if config_section_key == "data_sources":
                module_name_part = "data_handlers"
            elif config_section_key == "models":
                module_name_part = "model_wrappers"
            elif config_section_key == "judges":
                module_name_part = "judge_models"
            else:
                print(f"Error: Unknown config section '{config_section_key}' for dynamic import.")
                return None

            module = importlib.import_module(f"components.{module_name_part}")
            TargetClass = getattr(module, class_name)
            
            # TODO: let component classes to accept **kwargs to ignore unused ones.
            print(f"Attempting to create {class_name} with args: {init_args}")
            instance = TargetClass(**init_args)
            print(f"Successfully created instance of {class_name}.")
            return instance
        except ImportError as e:
            print(f"Error importing module for class {class_name}: {e}")
            return None
        except AttributeError as e:
            print(f"Error: Class {class_name} not found in module components.{module_name_part}: {e}")
            return None
        except TypeError as e:
            print(f"Error: Could not instantiate {class_name}. Check constructor arguments: {e}")
            print(f"Provided arguments: {init_args}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during instantiation of {class_name}: {e}")
            return None

# Example Usage (conceptual, typically called from main.py)
if __name__ == '__main__':

    sample_config_for_factory = {
        "base_config": {
            "random_seed": 10,
            "default_setting": "abc"
        },
        "data_sources": {
            "my_data_handler": {
                "class": "JsonlDataHandler", # Assuming this class exists in components.data_handlers
                "input_column": "q",
                "data_path": "/tmp/data.jsonl" # This would typically be overridden
            }
        },
        "models": {
            "my_model": {
                "class": "HuggingFaceCausalLM", # Assuming this class exists in components.model_wrappers
                "model_name_or_path": "test_model",
                "device": "cpu"
            }
        }
    }

    factory = ObjectFactory(sample_config_for_factory)

    # simulate CLI overrides for data_path
    data_handler_instance = factory.create("my_data_handler", "data_sources", data_path="/override/path/data.jsonl", sample_size=50)
    if data_handler_instance:
        print(f"Created data_handler: {type(data_handler_instance)}")
        # print(f"Data handler config: {data_handler_instance.__dict__}") # If it stores args

    model_instance = factory.create("my_model", "models", lora_adapter_path="/path/to/lora")
    if model_instance:
        print(f"Created model: {type(model_instance)}")
        # print(f"Model config: {model_instance.__dict__}")
