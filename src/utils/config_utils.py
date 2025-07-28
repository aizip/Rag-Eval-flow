import logging
from typing import Any, Dict, Optional

import yaml

# A type alias for a configuration dictionary to improve readability.
ConfigDict = Dict[str, Any]

# Configure a basic logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> Optional[ConfigDict]:
    """
    Loads and parses a YAML configuration file with robust error handling.

    Args:
        config_path: The full path to the YAML configuration file.

    Returns:
        A dictionary representing the YAML content, or None if an error occurs.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                logging.error(f"Configuration file {config_path} must be a dictionary.")
                return None
            return config_data
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {config_path}: {e}")
        return None

def get_config_value(config: ConfigDict, key_path: str, default: Any = None) -> Any:
    """
    Retrieves a value from a nested dictionary using a dot-separated key path.

    Args:
        config: The dictionary to search within.
        key_path: A dot-separated string of keys (e.g., "data.model.name").
        default: The value to return if the key path is not found.

    Returns:
        The value at the specified key path, or the default value.
    """
    keys = key_path.split('.')
    current_level = config
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return default
    return current_level

def recursive_update(base_dict: ConfigDict, new_dict: ConfigDict) -> ConfigDict:
    """
    Recursively updates a dictionary with values from another, modifying it in place.

    Args:
        base_dict: The dictionary to be updated.
        new_dict: The dictionary with new values to merge.

    Returns:
        The updated base_dict.
    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            recursive_update(base_dict[key], value)
        elif value is not None:
            base_dict[key] = value
    return base_dict