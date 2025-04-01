"""
================================================================================
Advanced Blocks Module (Dynamic _layer_configs Approach)
--------------------------------------------------------------------------------
This module defines advanced building blocks that use a dynamic layer-loading 
mechanism based on _layer_configs (as used in SimpleClassifier). Each block 
inherits from BaseModel so that its submodules are instantiated dynamically.

Included Blocks:
  • KernelMappingBlock:
      Approximates various kernel mappings (cosine, sigmoid, polynomial). All 
      possible kernel layers are listed, but only the one matching the
      initialization parameter (kernel_type) is active during forward.
  • AttentionBlock:
      A self-attention block.
  • ResidualBlock:
      A fully-connected residual unit.
================================================================================
"""

import math
import torch
import torch.nn as nn
from models.dl_models.base_model import BaseModel  # Ensure BaseModel implements dynamic _layer_configs

###############################################################################
# KernelMappingBlock using _layer_configs
###############################################################################
class KernelMappingBlock(BaseModel):
    """
    KernelMappingBlock approximates a kernel mapping using one of several potential
    transforms (cosine, sigmoid, polynomial). The desired kernel is selected at
    initialization by setting kernel_type.
    
    Dynamic Layer Definitions (via _layer_configs):
      • kernel_mapping_cosine: Instantiated if kernel_type == "cosine".
      • kernel_mapping_sigmoid: Instantiated if kernel_type == "sigmoid".
      • kernel_mapping_polynomial: Instantiated if kernel_type == "polynomial".
      
    Parameters:
      input_dim (int): Dimensionality of the input features.
      output_dim (int): Dimensionality of the transformed feature space.
      kernel_type (str): One of {"cosine", "sigmoid", "polynomial"}.
      dropout_rate (float): [Not used here; for compatibility]
      gamma (float): Kernel coefficient (default 1.0).
      degree (int): Degree for the polynomial kernel (only used if kernel_type == "polynomial").
    """
    _layer_configs = [
        {
            "name": "kernel_mapping_cosine",
            "layer_name": "KernelMappingCosine",
            "config": {
                "input_dim": lambda self: self.input_dim,
                "output_dim": lambda self: self.output_dim,
                "gamma": lambda self: self.gamma
            },
            "conditional": lambda self: self.kernel_type == "cosine"
        },
        {
            "name": "kernel_mapping_sigmoid",
            "layer_name": "KernelMappingSigmoid",
            "config": {
                "input_dim": lambda self: self.input_dim,
                "output_dim": lambda self: self.output_dim,
                "gamma": lambda self: self.gamma
            },
            "conditional": lambda self: self.kernel_type == "sigmoid"
        },
        {
            "name": "kernel_mapping_polynomial",
            "layer_name": "KernelMappingPolynomial",
            "config": {
                "input_dim": lambda self: self.input_dim,
                "output_dim": lambda self: self.output_dim,
                "gamma": lambda self: self.gamma,
                "degree": lambda self: self.degree
            },
            "conditional": lambda self: self.kernel_type == "polynomial"
        },
    ]
    
    def __init__(self, input_dim: int, output_dim: int, kernel_type: str = "cosine",
                 gamma: float = 1.0, degree: int = 2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.degree = degree  # Only used for polynomial kernel.
        super(KernelMappingBlock, self).__init__()
    
    def forward(self, x):
        # Only one of the kernel mapping layers is instantiated due to the "conditional" lambdas.
        # Forward through the active kernel mapping.
        if self.kernel_type == "cosine" and hasattr(self, "kernel_mapping_cosine"):
            return self.kernel_mapping_cosine(x)
        elif self.kernel_type == "sigmoid" and hasattr(self, "kernel_mapping_sigmoid"):
            return self.kernel_mapping_sigmoid(x)
        elif self.kernel_type == "polynomial" and hasattr(self, "kernel_mapping_polynomial"):
            return self.kernel_mapping_polynomial(x)
        else:
            raise ValueError(f"Invalid kernel_type '{self.kernel_type}' selected.")

###############################################################################
# AttentionBlock using _layer_configs
###############################################################################
class AttentionBlock(BaseModel):
    """
    AttentionBlock implements a self-attention mechanism using dynamic layer 
    configuration. It creates the query, key, and value linear layers via _layer_configs.
    
    Parameters:
      input_dim (int): Dimensionality of the input features.
      attention_dim (int): Dimensionality of the attention space.
    """
    _layer_configs = [
        {
            "name": "query",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.input_dim,
                "out_features": lambda self: self.attention_dim
            }
        },
        {
            "name": "key",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.input_dim,
                "out_features": lambda self: self.attention_dim
            }
        },
        {
            "name": "value",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.input_dim,
                "out_features": lambda self: self.input_dim
            }
        }
    ]
    
    def __init__(self, input_dim: int, attention_dim: int):
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        # Compute the scaling factor for attention.
        self.scale = attention_dim ** -0.5
        super(AttentionBlock, self).__init__()
    
    def forward(self, x):
        q = self.query(x)  # shape: (batch, attention_dim)
        k = self.key(x)    # shape: (batch, attention_dim)
        attn_weights = torch.softmax(q * self.scale, dim=-1)  # shape: (batch, attention_dim)
        weighted = self.value(x) * attn_weights.unsqueeze(-1).squeeze()
        return weighted + x  # Residual connection

###############################################################################
# ResidualBlock using _layer_configs
###############################################################################
class ResidualBlock(BaseModel):
    """
    ResidualBlock implements a fully-connected residual unit.
    It creates its submodules dynamically and includes an optional residual mapping
    if the input and hidden dimensions differ.
    
    Parameters:
      input_dim (int): Input feature dimension.
      hidden_dim (int): Hidden layer dimension.
      dropout_rate (float): Dropout probability.
    """
    _layer_configs = [
        {
            "name": "fc",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.input_dim,
                "out_features": lambda self: self.hidden_dim
            }
        },
        {
            "name": "bn",
            "layer_name": "BatchNorm1d",
            "config": {
                "num_features": lambda self: self.hidden_dim
            }
        },
        {
            "name": "relu",
            "layer_name": "ReLU",
            "config": {
                "inplace": True
            }
        },
        {
            "name": "dropout",
            "layer_name": "Dropout",
            "config": {
                "p": lambda self: self.dropout_rate
            }
        },
        {
            "name": "residual_mapping",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.input_dim,
                "out_features": lambda self: self.hidden_dim
            },
            "conditional": lambda self: self.input_dim != self.hidden_dim
        }
    ]
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        super(ResidualBlock, self).__init__()
    
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        # If residual_mapping was instantiated, apply it to the input.
        if hasattr(self, "residual_mapping"):
            identity = self.residual_mapping(identity)
        return out + identity

###############################################################################
# Testing the blocks
###############################################################################
if __name__ == "__main__":
    import torch
    
    # Test KernelMappingBlock with different kernel types.
    for kernel in ["cosine", "sigmoid", "polynomial"]:
        print(f"\nTesting kernel type: {kernel}")
        if kernel == "polynomial":
            km_block = KernelMappingBlock(input_dim=10, output_dim=64, kernel_type=kernel, gamma=1.0, degree=3)
        else:
            km_block = KernelMappingBlock(input_dim=10, output_dim=64, kernel_type=kernel, gamma=1.0)
        x_dummy = torch.randn(5, 10)
        km_out = km_block(x_dummy)
        print(f"Output shape for {kernel} kernel mapping:", km_out.shape)
    
    # Test AttentionBlock.
    attn_block = AttentionBlock(input_dim=64, attention_dim=32)
    attn_out = attn_block(km_out)
    print("AttentionBlock output shape:", attn_out.shape)
    
    # Test ResidualBlock.
    res_block = ResidualBlock(input_dim=64, hidden_dim=64, dropout_rate=0.3)
    res_out = res_block(km_out)
    print("ResidualBlock output shape:", res_out.shape)