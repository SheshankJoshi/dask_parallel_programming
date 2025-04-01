"""
Layer Map Extended Module
---------------------------
This module defines an extended mapping of torch.nn layers and blocks that are not
included in layer_map_basic.py. The EXTENDED_LAYER_MAPPING dictionary contains additional
layers such as transposed convolutions, normalization layers, advanced activations, recurrent cell
variants, transformer modules, and container modules.

Each dictionary entry contains:
  - "class": The torch.nn layer or block class.
  - "required_params": List of parameters required for instantiation.
  - "optional_params": Dictionary of optional parameters with default values.

Primary Function:
  create_layer_extended(layer_name: str, config: dict) -> nn.Module
    Instantiates and returns a layer using the extended mapping.

Example:
  >>> extended_layer = create_layer_extended("LayerNorm", {"normalized_shape": [128]})
"""

import torch.nn as nn

EXTENDED_LAYER_MAPPING = {
    # Additional Convolutional Layers
    "ConvTranspose1d": {
        "class": nn.ConvTranspose1d,
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": {
            "stride": 1,
            "padding": 0,
            "output_padding": 0,
            "groups": 1,
            "bias": True,
            "dilation": 1,
            "padding_mode": 'zeros'
        },
    },
    "ConvTranspose2d": {
        "class": nn.ConvTranspose2d,
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": {
            "stride": 1,
            "padding": 0,
            "output_padding": 0,
            "groups": 1,
            "bias": True,
            "dilation": 1,
            "padding_mode": 'zeros'
        },
    },
    "ConvTranspose3d": {
        "class": nn.ConvTranspose3d,
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": {
            "stride": 1,
            "padding": 0,
            "output_padding": 0,
            "groups": 1,
            "bias": True,
            "dilation": 1,
            "padding_mode": 'zeros'
        },
    },
    "LayerNorm": {
        "class": nn.LayerNorm,
        "required_params": ["normalized_shape"],
        "optional_params": {
            "eps": 1e-05,
            "elementwise_affine": True
        },
    },
    "InstanceNorm1d": {
        "class": nn.InstanceNorm1d,
        "required_params": ["num_features"],
        "optional_params": {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": False,
            "track_running_stats": False
        },
    },
    "InstanceNorm2d": {
        "class": nn.InstanceNorm2d,
        "required_params": ["num_features"],
        "optional_params": {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": False,
            "track_running_stats": False
        },
    },
    "InstanceNorm3d": {
        "class": nn.InstanceNorm3d,
        "required_params": ["num_features"],
        "optional_params": {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": False,
            "track_running_stats": False
        },
    },
    "GroupNorm": {
        "class": nn.GroupNorm,
        "required_params": ["num_groups", "num_channels"],
        "optional_params": {
            "eps": 1e-05,
            "affine": True
        },
    },
    "LocalResponseNorm": {
        "class": nn.LocalResponseNorm,
        "required_params": ["size"],
        "optional_params": {
            "alpha": 1e-4,
            "beta": 0.75,
            "k": 1.0
        },
    },
    "Softmax": {
        "class": nn.Softmax,
        "required_params": ["dim"],
        "optional_params": {},
    },
    "Softmax2d": {
        "class": nn.Softmax2d,
        "required_params": [],
        "optional_params": {},
    },
    "LogSoftmax": {
        "class": nn.LogSoftmax,
        "required_params": ["dim"],
        "optional_params": {},
    },
    "PReLU": {
        "class": nn.PReLU,
        "required_params": [],
        "optional_params": {"num_parameters": 1, "init": 0.25},
    },
    "SELU": {
        "class": nn.SELU,
        "required_params": [],
        "optional_params": {"inplace": False},
    },
    "Hardshrink": {
        "class": nn.Hardshrink,
        "required_params": [],
        "optional_params": {"lambd": 0.5},
    },
    "Hardtanh": {
        "class": nn.Hardtanh,
        "required_params": [],
        "optional_params": {"min_val": -1.0, "max_val": 1.0, "inplace": False},
    },
    "Sigmoid": {
        "class": nn.Sigmoid,
        "required_params": [],
        "optional_params": {},
    },
    "Tanh": {
        "class": nn.Tanh,
        "required_params": [],
        "optional_params": {},
    },
    "Softplus": {
        "class": nn.Softplus,
        "required_params": [],
        "optional_params": {"beta": 1, "threshold": 20},
    },
    "Softsign": {
        "class": nn.Softsign,
        "required_params": [],
        "optional_params": {},
    },
    "RNN": {
        "class": nn.RNN,
        "required_params": ["input_size", "hidden_size"],
        "optional_params": {
            "num_layers": 1,
            "nonlinearity": "tanh",
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False
        },
    },
    "RNNCell": {
        "class": nn.RNNCell,
        "required_params": ["input_size", "hidden_size"],
        "optional_params": {"bias": True},
    },
    "LSTMCell": {
        "class": nn.LSTMCell,
        "required_params": ["input_size", "hidden_size"],
        "optional_params": {"bias": True},
    },
    "GRUCell": {
        "class": nn.GRUCell,
        "required_params": ["input_size", "hidden_size"],
        "optional_params": {"bias": True},
    },
    "Upsample": {
        "class": nn.Upsample,
        "required_params": [],
        "optional_params": {
            "size": None,
            "scale_factor": None,
            "mode": "nearest",
            "align_corners": None
        },
    },
    "Transformer": {
        "class": nn.Transformer,
        "required_params": [],
        "optional_params": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu"
        },
    },
    "MultiheadAttention": {
        "class": nn.MultiheadAttention,
        "required_params": ["embed_dim", "num_heads"],
        "optional_params": {"dropout": 0.0, "bias": True, "add_bias_kv": False, "add_zero_attn": False},
    },
    "TransformerEncoderLayer": {
        "class": nn.TransformerEncoderLayer,
        "required_params": ["d_model", "nhead"],
        "optional_params": {
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu",
            "layer_norm_eps": 1e-05
        },
    },
    "TransformerEncoder": {
        "class": nn.TransformerEncoder,
        "required_params": ["encoder_layer", "num_layers"],
        "optional_params": {"norm": None},
    },
    "TransformerDecoderLayer": {
        "class": nn.TransformerDecoderLayer,
        "required_params": ["d_model", "nhead"],
        "optional_params": {
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu",
            "layer_norm_eps": 1e-05
        },
    },
    "TransformerDecoder": {
        "class": nn.TransformerDecoder,
        "required_params": ["decoder_layer", "num_layers"],
        "optional_params": {"norm": None},
    },
    "ModuleList": {
        "class": nn.ModuleList,
        "required_params": ["modules"],
        "optional_params": {},
    },
    "ModuleDict": {
        "class": nn.ModuleDict,
        "required_params": ["modules"],
        "optional_params": {},
    },
    "Sequential": {
        "class": nn.Sequential,
        "required_params": ["args"],
        "optional_params": {},
    },
    "ResNetBlock": {
        "class": nn.Identity,  # Placeholder; replace with an actual block if needed
        "required_params": [],
        "optional_params": {},
    },
    "InceptionModule": {
        "class": nn.Identity,  # Placeholder; replace as needed
        "required_params": [],
        "optional_params": {},
    },
}

def create_layer_extended(layer_name: str, config: dict) -> nn.Module:
    """
    Instantiate and return a torch.nn layer from EXTENDED_LAYER_MAPPING using the provided configuration.
    If layer_name is not found in EXTENDED_LAYER_MAPPING, then check in CUSTOM_LAYER_MAPPING from custom_layers.py.

    Parameters:
      layer_name (str): Key representing the layer.
      config (dict): Configuration dictionary containing required and optional parameters.
    
    Returns:
      nn.Module: Instantiated layer.
    
    Raises:
      ValueError: If layer_name is not defined in either EXTENDED_LAYER_MAPPING or CUSTOM_LAYER_MAPPING,
                  or if a required parameter is missing.
    """
    mapping = None
    if layer_name in EXTENDED_LAYER_MAPPING:
        mapping = EXTENDED_LAYER_MAPPING
    else:
        try:
            from managers.custom_layers import CUSTOM_LAYER_MAPPING  # adjust the import path as needed
        except ImportError as e:
            raise ImportError("Could not import custom_layers module. Ensure it exists and is on the PYTHONPATH.") from e
        if layer_name in CUSTOM_LAYER_MAPPING:
            mapping = CUSTOM_LAYER_MAPPING
        else:
            raise ValueError(f"Layer {layer_name} is not defined in EXTENDED_LAYER_MAPPING or CUSTOM_LAYER_MAPPING.")
    
    layer_info = mapping[layer_name]
    layer_class = layer_info["class"]
    instance_params = {}
    for param in layer_info.get("required_params", []):
        if param not in config:
            raise ValueError(f"Missing required parameter '{param}' for layer '{layer_name}'.")
        instance_params[param] = config[param]
    for param, default_val in layer_info.get("optional_params", {}).items():
        instance_params[param] = config.get(param, default_val)
    return layer_class(**instance_params)
