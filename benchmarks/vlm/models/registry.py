"""
Model registry - maps model names to their wrapper classes.
Add new models here as they are implemented.
"""

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model wrapper class."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name, **kwargs):
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found. Available: {available}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())


# Import all model implementations to trigger registration
from . import qwen2_vl
from . import llava
from . import internvl
from . import openai_api
from . import llava_v15
from . import gemma3
from . import qwen3_5  # Qwen3-VL

from . import phi4
