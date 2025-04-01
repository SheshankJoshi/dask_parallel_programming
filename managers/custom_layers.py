import torch
import torch.nn as nn
import math

class KernelMappingCosine(nn.Module):
    """
    KernelMappingCosine approximates an RBF kernel using Random Fourier Features
    with a cosine transformation.
    
    Parameters:
      input_dim (int): Dimensionality of the input features.
      output_dim (int): Dimensionality of the transformed feature space.
      gamma (float): Kernel coefficient for RBF (default is 1.0).
    """
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 1.0):
        super(KernelMappingCosine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * math.sqrt(2 * gamma))
        self.b = nn.Parameter(2 * math.pi * torch.rand(output_dim))

    def forward(self, x):
        projection = x @ self.W + self.b
        return math.sqrt(2.0 / self.output_dim) * torch.cos(projection)

class KernelMappingSigmoid(nn.Module):
    """
    KernelMappingSigmoid computes a kernel mapping based on a sigmoid activation.
    It uses random weights and biases to project the input and then applies a sigmoid.
    
    Parameters:
      input_dim (int): Dimensionality of the input.
      output_dim (int): Dimensionality of the transformed space.
      gamma (float): Scaling coefficient (default is 1.0).
    """
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 1.0):
        super(KernelMappingSigmoid, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * math.sqrt(2 * gamma))
        self.b = nn.Parameter(2 * math.pi * torch.rand(output_dim))

    def forward(self, x):
        projection = x @ self.W + self.b
        return torch.sigmoid(projection)

class KernelMappingPolynomial(nn.Module):
    """
    KernelMappingPolynomial approximates a polynomial kernel mapping.
    It computes (xW + b)^degree.
    
    Parameters:
      input_dim (int): Dimensionality of the input.
      output_dim (int): Dimensionality of the transformed space.
      degree (int): Degree of the polynomial (default is 2).
      gamma (float): Scaling coefficient (default is 1.0).
    """
    def __init__(self, input_dim: int, output_dim: int, degree: int = 2, gamma: float = 1.0):
        super(KernelMappingPolynomial, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.gamma = gamma
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * math.sqrt(2 * gamma))
        self.b = nn.Parameter(2 * math.pi * torch.rand(output_dim))

    def forward(self, x):
        projection = x @ self.W + self.b
        return torch.pow(projection, self.degree)

# CUSTOM_LAYER_MAPPING now includes only the kernel mapping layers.
CUSTOM_LAYER_MAPPING = {
    "KernelMappingCosine": {
        "class": KernelMappingCosine,
        "required_params": ["input_dim", "output_dim"],
        "optional_params": {
            "gamma": 1.0
        },
    },
    "KernelMappingSigmoid": {
        "class": KernelMappingSigmoid,
        "required_params": ["input_dim", "output_dim"],
        "optional_params": {
            "gamma": 1.0
        },
    },
    "KernelMappingPolynomial": {
        "class": KernelMappingPolynomial,
        "required_params": ["input_dim", "output_dim", "degree"],
        "optional_params": {
            "gamma": 1.0
        },
    }
}

def create_layer_custom(layer_name: str, config: dict) -> nn.Module:
    """
    Instantiate and return a torch.nn layer from CUSTOM_LAYER_MAPPING using the provided configuration.
    
    Parameters:
      layer_name (str): Key representing the custom layer.
      config (dict): Configuration dictionary containing required and optional parameters.
      
    Returns:
      nn.Module: Instantiated custom layer.
    
    Raises:
      ValueError: If layer_name is not defined in CUSTOM_LAYER_MAPPING or if a required parameter is missing.
    """
    if layer_name not in CUSTOM_LAYER_MAPPING:
        raise ValueError(f"Layer {layer_name} is not defined in CUSTOM_LAYER_MAPPING.")
    layer_info = CUSTOM_LAYER_MAPPING[layer_name]
    layer_class = layer_info["class"]
    instance_params = {}
    for param in layer_info.get("required_params", []):
        if param not in config:
            raise ValueError(f"Missing required parameter '{param}' for custom layer '{layer_name}'.")
        instance_params[param] = config[param]
    for param, default_val in layer_info.get("optional_params", {}).items():
        instance_params[param] = config.get(param, default_val)
    return layer_class(**instance_params)

