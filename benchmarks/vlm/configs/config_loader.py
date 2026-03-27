"""
Config loader with variable interpolation.
Loads base.yaml + dataset config + model config and merges them.
"""
import os
import re
import yaml
from copy import deepcopy


def _interpolate(obj, context):
    """Recursively interpolate ${var.path} references in config values."""
    if isinstance(obj, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, obj)
        for match in matches:
            keys = match.split('.')
            val = context
            for k in keys:
                val = val[k]
            obj = obj.replace(f'${{{match}}}', str(val))
        return obj
    elif isinstance(obj, dict):
        return {k: _interpolate(v, context) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate(v, context) for v in obj]
    return obj


def _deep_merge(base, override):
    """Deep merge override into base dict."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(dataset_variant, model_name=None):
    """
    Load merged config for a dataset variant and optional model.

    Args:
        dataset_variant: one of 'original', 'weather', 'night', 'small'
        model_name: optional model config name (e.g., 'qwen2_vl_7b')

    Returns:
        dict: merged configuration
    """
    config_dir = os.path.dirname(os.path.abspath(__file__))

    # Load base
    with open(os.path.join(config_dir, 'base.yaml')) as f:
        config = yaml.safe_load(f)

    # Load dataset config
    dataset_file = os.path.join(config_dir, f'dataset_{dataset_variant}.yaml')
    if os.path.exists(dataset_file):
        with open(dataset_file) as f:
            dataset_cfg = yaml.safe_load(f)
        config = _deep_merge(config, dataset_cfg)
    else:
        raise FileNotFoundError(f"Dataset config not found: {dataset_file}")

    # Load model config if specified
    if model_name:
        model_file = os.path.join(config_dir, 'models', f'{model_name}.yaml')
        if os.path.exists(model_file):
            with open(model_file) as f:
                model_cfg = yaml.safe_load(f)
            config = _deep_merge(config, model_cfg)
        else:
            raise FileNotFoundError(f"Model config not found: {model_file}")

    # Interpolate variables
    config = _interpolate(config, config)

    return config
