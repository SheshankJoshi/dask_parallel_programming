"""
================================================================================
ComplexClassifier Module
--------------------------------------------------------------------------------
This module defines an advanced fully-connected neural network classifier that
extends the basic SimpleClassifier. It is designed to provide a generic, yet
complicated classifier architecture with support for:
  - Deep architectures with multiple hidden layers.
  - Optional kernel mapping using Random Fourier Features (RBF kernel approximation)
    to approximate kernel methods.
  - Batch normalization and dropout for improved training stability and regularization.
  - Dynamically loaded layers via a configuration mechanism (inherited from BaseModel).

Example Architecture:
  [Optional] KernelMappingBlock --> fc1 --> bn1 --> ReLU --> Dropout -->
                                  fc2 --> bn2 --> ReLU --> Dropout -->
                                  fc3 --> (output logits)

Usage Example:
  >>> model = ComplexClassifier(input_size=10, hidden_size=64, num_classes=2,
  ...                            use_kernel=True, kernel_dim=128, dropout_rate=0.5)
  >>> print(model)
  >>> x = torch.randn(batch_size, input_size)
  >>> out = model(x)
  >>> print("Output shape:", out.shape)

Dependencies:
  - Inherits dynamic layer loading behavior from models/dl_models/base_model.py.
  - Uses KernelMappingBlock from the building_blocks package for kernel mapping.
================================================================================
"""

import math
import torch
import torch.nn as nn
from models.dl_models.base_model import BaseModel  # Assumes BaseModel implements dynamic layer-loading
# Use the KernelMappingBlock from the building_blocks package instead of a local KernelMapping.
from models.building_blocks.advanced_blocks import KernelMappingBlock

class ComplexClassifier(BaseModel):
    """
    ComplexClassifier implements an advanced fully-connected neural network with:
      - (Optional) A kernel mapping block to approximate kernel methods.
      - Multiple fully-connected layers with batch normalization and dropout.
      - Non-linear activations (ReLU).

    Dynamic Layer Definitions (via _layer_configs):
      • kernel_mapping (optional): Uses KernelMappingBlock if use_kernel is True.
      • fc1: First Linear layer from effective input size to hidden_size.
      • bn1: Batch normalization on the first hidden layer.
      • activation1: ReLU activation.
      • dropout1: Dropout layer with specified dropout_rate.
      • fc2: Second Linear layer mapping hidden_size to hidden_size.
      • bn2: Batch normalization on the second hidden layer.
      • activation2: ReLU activation.
      • dropout2: Dropout layer with specified dropout_rate.
      • fc3: Final Linear layer mapping hidden_size to num_classes.
      
    Parameters:
      input_size   (int): Dimensionality of the raw input.
      hidden_size  (int): Number of neurons in hidden layers.
      num_classes  (int): Number of output classes.
      use_kernel   (bool): Whether to apply kernel mapping via Random Fourier Features.
      kernel_dim   (int): Dimension for the kernel mapping (if used).
      dropout_rate (float): Dropout probability.
      
    Note:
      When use_kernel is True, the effective input to the network becomes kernel_dim.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 use_kernel: bool = False, kernel_dim: int = 64, dropout_rate: float = 0.5):
        # Store initialization parameters.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_kernel = use_kernel
        self.kernel_dim = kernel_dim
        self.dropout_rate = dropout_rate
        # Determine effective input dimension.
        self.effective_input = kernel_dim if use_kernel else input_size
        super(ComplexClassifier, self).__init__()

    # _layer_configs defines layers dynamically through configuration dictionaries.
    _layer_configs = [
        {
            "name": "kernel_mapping",
            "layer_name": "KernelMappingBlock",  # Maps to KernelMappingBlock from the building_blocks package.
            "config": {
                "input_dim": lambda self: self.input_size,
                "output_dim": lambda self: self.kernel_dim,
                "gamma": lambda self: 1.0
            },
            "conditional": lambda self: self.use_kernel
        },
        {
            "name": "fc1",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.effective_input,
                "out_features": lambda self: self.hidden_size
            }
        },
        {
            "name": "bn1",
            "layer_name": "BatchNorm1d",
            "config": {
                "num_features": lambda self: self.hidden_size
            }
        },
        {
            "name": "activation1",
            "layer_name": "ReLU",
            "config": {
                "inplace": True
            }
        },
        {
            "name": "dropout1",
            "layer_name": "Dropout",
            "config": {
                "p": lambda self: self.dropout_rate
            }
        },
        {
            "name": "fc2",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.hidden_size,
                "out_features": lambda self: self.hidden_size
            }
        },
        {
            "name": "bn2",
            "layer_name": "BatchNorm1d",
            "config": {
                "num_features": lambda self: self.hidden_size
            }
        },
        {
            "name": "activation2",
            "layer_name": "ReLU",
            "config": {
                "inplace": True
            }
        },
        {
            "name": "dropout2",
            "layer_name": "Dropout",
            "config": {
                "p": lambda self: self.dropout_rate
            }
        },
        {
            "name": "fc3",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.hidden_size,
                "out_features": lambda self: self.num_classes
            }
        }
    ]
    
    def forward(self, x):
        """
        Forward pass for ComplexClassifier.
        If kernel mapping is enabled, first transform the inputs using the kernel_mapping block.
        Then apply deep fully-connected layers interleaved with batch normalization,
        ReLU activations, and dropout for regularization.

        Parameters:
          x (Tensor): Input tensor with shape (batch_size, input_size)

        Returns:
          Tensor: Logits with shape (batch_size, num_classes)
        """
        if self.use_kernel:
            x = self.kernel_mapping(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    import torch
    import torch.optim as optim
    import torch.nn as nn

    # Example usage of the ComplexClassifier:
    # Create an instance with kernel mapping enabled.
    model = ComplexClassifier(input_size=10, hidden_size=64, num_classes=2,
                              use_kernel=True, kernel_dim=128, dropout_rate=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("ComplexClassifier structure:")
    print(model)

    # Create a dummy input tensor.
    x = torch.randn(5, 10)  # (batch_size, input_size)
    # Forward pass.
    out = model(x)
    print("Output shape:", out.shape)