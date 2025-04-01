"""
Layer Map Basic Module
------------------------
This module defines a basic mapping of torch.nn layers. The mapping, stored in the
LAYER_MAPPING dictionary, associates string keys to configuration dictionaries used for
instantiating torch.nn layers.

Each configuration dictionary contains:
  - "class": The torch.nn layer class.
  - "required_params": List of parameters required during initialization.
  - "optional_params": Dictionary of optional parameters with default values.

Primary Function:
  create_layer(layer_name: str, config: dict) -> nn.Module
    Instantiates and returns a torch.nn layer given a key and a configuration dictionary.

Example:
  >>> layer = create_layer("Linear", {"in_features": 10, "out_features": 5})
"""

import torch.nn as nn
LAYER_MAPPING = {
    # Linear layer
    "Linear": {
        "class": nn.Linear,
        "required_params": ["in_features", "out_features"],
        "optional_params": {"bias": True},
    },
    # Convolution layers
    "Conv1d": {
        "class": nn.Conv1d,
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": {
            "stride": 1, 
            "padding": 0, 
            "dilation": 1, 
            "groups": 1, 
            "bias": True, 
            "padding_mode": "zeros"
        },
    },
    "Conv2d": {
        "class": nn.Conv2d,
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": {
            "stride": 1, 
            "padding": 0, 
            "dilation": 1, 
            "groups": 1, 
            "bias": True, 
            "padding_mode": "zeros"
        },
    },
    "Conv3d": {
        "class": nn.Conv3d,
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": {
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros"
        },
    },
    # Pooling layers
    "MaxPool1d": {
        "class": nn.MaxPool1d,
        "required_params": ["kernel_size"],
        "optional_params": {
            "stride": None,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False
        },
    },
    "MaxPool2d": {
        "class": nn.MaxPool2d,
        "required_params": ["kernel_size"],
        "optional_params": {
            "stride": None,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False
        },
    },
    "AvgPool1d": {
        "class": nn.AvgPool1d,
        "required_params": ["kernel_size"],
        "optional_params": {
            "stride": None,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True
        },
    },
    "AvgPool2d": {
        "class": nn.AvgPool2d,
        "required_params": ["kernel_size"],
        "optional_params": {
            "stride": None,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True
        },
    },
    # Normalization layers
    "BatchNorm1d": {
        "class": nn.BatchNorm1d,
        "required_params": ["num_features"],
        "optional_params": {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True
        },
    },
    "BatchNorm2d": {
        "class": nn.BatchNorm2d,
        "required_params": ["num_features"],
        "optional_params": {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True
        },
    },
    "BatchNorm3d": {
        "class": nn.BatchNorm3d,
        "required_params": ["num_features"],
        "optional_params": {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True
        },
    },
    # Activation layers
    "ReLU": {
        "class": nn.ReLU,
        "required_params": [],
        "optional_params": {"inplace": False},
    },
    "LeakyReLU": {
        "class": nn.LeakyReLU,
        "required_params": [],
        "optional_params": {"negative_slope": 0.01, "inplace": False},
    },
    "ELU": {
        "class": nn.ELU,
        "required_params": [],
        "optional_params": {"alpha": 1.0, "inplace": False},
    },
    # Dropout layers
    "Dropout": {
        "class": nn.Dropout,
        "required_params": [],
        "optional_params": {"p": 0.5, "inplace": False},
    },
    "Dropout2d": {
        "class": nn.Dropout2d,
        "required_params": [],
        "optional_params": {"p": 0.5, "inplace": False},
    },
    # Recurrent layers
    "LSTM": {
        "class": nn.LSTM,
        "required_params": ["input_size", "hidden_size"],
        "optional_params": {
            "num_layers": 1,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False
        },
    },
    "GRU": {
        "class": nn.GRU,
        "required_params": ["input_size", "hidden_size"],
        "optional_params": {
            "num_layers": 1,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False
        },
    },
    # Embedding layer
    "Embedding": {
        "class": nn.Embedding,
        "required_params": ["num_embeddings", "embedding_dim"],
        "optional_params": {
            "padding_idx": None,
            "max_norm": None,
            "norm_type": 2.0,
            "scale_grad_by_freq": False,
            "sparse": False
        },
    },
}

def create_layer(layer_name: str, config: dict) -> nn.Module:
    """
    Create and return an instance of a layer using the LAYER_MAPPING definition.
    
    Parameters:
      layer_name (str): The key representing the layer in LAYER_MAPPING.
      config (dict): Dictionary containing the parameters required to create the layer.
                     It must include all keys in the "required_params" list and may
                     include any overrides for "optional_params".
                     
    Returns:
      nn.Module: An instantiated layer.
    
    Raises:
      ValueError: If the layer_name is not available in the mapping or 
                  if a required parameter is missing in the provided config.
    """
    if layer_name not in LAYER_MAPPING:
        raise ValueError(f"Layer {layer_name} is not defined in the LAYER_MAPPING.")
    
    layer_info = LAYER_MAPPING[layer_name]
    layer_class = layer_info["class"]
    instance_params = {}
    
    # Ensure all required parameters are provided
    for param in layer_info.get("required_params", []):
        if param not in config:
            raise ValueError(f"Missing required parameter '{param}' for layer '{layer_name}'.")
        instance_params[param] = config[param]
    
    # Process optional parameters (use provided values or fall back to defaults)
    for param, default_val in layer_info.get("optional_params", {}).items():
        instance_params[param] = config.get(param, default_val)
    
    return layer_class(**instance_params)
    
# Example usage:
if __name__ == "__main__":
    # Define a list of layer configurations
    layer_configs = [
        {"layer_name": "Linear", "config": {"in_features": 128, "out_features": 64}},
        {"layer_name": "ReLU", "config": {"inplace": True}},
        {"layer_name": "Dropout", "config": {"p": 0.3}},
        {"layer_name": "Linear", "config": {"in_features": 64, "out_features": 10}},
    ]

    # Build the model dynamically based on the layer configurations
    layers = []
    for layer_def in layer_configs:
        layer = create_layer(layer_def["layer_name"], layer_def["config"])
        layers.append(layer)
    
    # Using nn.Sequential to chain the layers together
    model = nn.Sequential(*layers)
    print(model)