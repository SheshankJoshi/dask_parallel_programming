"""
Prebuilt Sequential Model Mapping Module (Time Series Models)
----------------------------------------------------
This module provides a registry for prebuilt sequential (time series) models.
The models (e.g., SimpleLSTMForecast, SimpleGRUForecast, SimpleTransformerTimeSeries) are defined in
the models/dl_models/timeseries_models.py module.

The mapping dictionary, SEQ_MODEL_MAPPING, defines for each model:
  - "class": The model class.
  - "required_params": A list of parameters that must be provided at initialization.
  - "optional_params": A dictionary of optional parameters with default values.

Primary Function:
  create_seq_model(model_name: str, config: dict) -> nn.Module
    Instantiates and returns a time series model using the configuration provided.

Usage Example:
  >>> model = create_seq_model("SimpleLSTMForecast", {"input_size": 10, "hidden_size": 32, "num_layers": 2, "output_size": 1})
"""

import torch
import torch.nn as nn
from models.dl_models.timeseries_models import SimpleLSTMForecast, SimpleGRUForecast, SimpleTransformerTimeSeries

SEQ_MODEL_MAPPING = {
    "SimpleLSTMForecast": {
        "class": SimpleLSTMForecast,
        "required_params": ["input_size", "hidden_size", "num_layers", "output_size"],
        "optional_params": {"dropout": 0.0},
    },
    "SimpleGRUForecast": {
        "class": SimpleGRUForecast,
        "required_params": ["input_size", "hidden_size", "num_layers", "output_size"],
        "optional_params": {"dropout": 0.0},
    },
    "SimpleTransformerTimeSeries": {
        "class": SimpleTransformerTimeSeries,
        "required_params": ["input_size", "num_heads", "num_encoder_layers", "output_size"],
        "optional_params": {"hidden_dim": 256, "dropout": 0.1},
    },
}

def create_seq_model(model_name: str, config: dict) -> nn.Module:
    """
    Instantiate and return a sequential (time series) model from SEQ_MODEL_MAPPING.

    Parameters:
      model_name (str): Key identifying the model.
      config (dict): Dictionary containing model configuration parameters.
                     Must include all required parameters.

    Returns:
      nn.Module: Instantiated model.
    
    Raises:
      ValueError: If model_name is not defined in SEQ_MODEL_MAPPING or if any required parameter is missing.
    """
    if model_name not in SEQ_MODEL_MAPPING:
        raise ValueError(f"Model {model_name} is not defined in SEQ_MODEL_MAPPING.")
    model_info = SEQ_MODEL_MAPPING[model_name]
    model_class = model_info["class"]
    instance_params = {}
    for param in model_info.get("required_params", []):
        if param not in config:
            raise ValueError(f"Missing required parameter '{param}' for model '{model_name}'.")
        instance_params[param] = config[param]
    for param, default_val in model_info.get("optional_params", {}).items():
        instance_params[param] = config.get(param, default_val)
    return model_class(**instance_params)