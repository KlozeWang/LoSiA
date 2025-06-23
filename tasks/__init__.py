import importlib
from typing import Dict, Any

def get_preprocessor(module_name: str):
    try:
        module = importlib.import_module(f"tasks.{module_name}")
        return module.preprocess_function
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Error occurred when loading preprocessing function for '{module_name}': {str(e)}")