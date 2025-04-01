from abc import ABC, abstractmethod
import torch.nn as nn
from managers.layer_map_basic import create_layer as create_layer_basic
from managers.layer_map_extended import create_layer_extended

class BaseModel(nn.Module, ABC):
    # Subclasses must define _layer_configs as a class variable.
    # Each entry in _layer_configs should be a dict with:
    #   - "name": the attribute name to assign for the layer.
    #   - "layer_name": the key to look up in LAYER_MAPPING (either basic or extended).
    #   - "config": a dict of configuration parameters. Values can be callables
    #               that accept self (the instance) so that runtime parameters can be injected.
    _layer_configs = []  # Default empty list; override in subclass.

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()
        # Build layers as attributes from _layer_configs.
        for layer_config in self.__class__._layer_configs:
            if "name" not in layer_config:
                raise ValueError("Each layer config must have a 'name' key to assign an attribute.")
            # Evaluate configuration parameters if they are callables.
            evaluated_config = {}
            for key, val in layer_config["config"].items():
                evaluated_config[key] = val(self) if callable(val) else val

            # Try to create the layer using the basic mapping. If not found, try extended.
            try:
                layer_instance = create_layer_basic(layer_config["layer_name"], evaluated_config)
            except ValueError:
                layer_instance = create_layer_extended(layer_config["layer_name"], evaluated_config)
            setattr(self, layer_config["name"], layer_instance)

    @abstractmethod
    def forward(self, x):
        pass