"""
================================================================================
AdvancedClassifier Module
--------------------------------------------------------------------------------
This module defines an advanced classifier designed to address more complicated 
modeling challenges beyond a standard fully-connected network. "AdvancedClassifier" 
incorporates the following features:

  - Deep multi-layer architecture with residual (skip) connections.
  - Self-attention block to re-weight features dynamically.
  - Optional kernel mapping can be enabled separately if required.
  - Batch normalization, dropout and ReLU activations for improved training
    stability and regularization.
  - Dynamically loaded layers via a configuration mechanism from BaseModel.

Architecture Example:
         Input --> [ResBlock] --> AttentionBlock --> fc1 --> bn1 --> ReLU --> 
         Dropout --> fc2 --> [Residual Connection] --> fc3 --> Output logits

Usage Example:
  >>> from models.dl_models.advanced_classifier import AdvancedClassifier
  >>> model = AdvancedClassifier(input_size=10, hidden_size=64, num_classes=2, attention_dim=32, num_res_blocks=2, dropout_rate=0.5)
  >>> print(model)
  >>> x = torch.randn(batch_size, input_size)
  >>> out = model(x)
  >>> print("Output shape:", out.shape)

Dependencies:
  - Inherits dynamic layer loading from models.dl_models.base_model.BaseModel.
  - Uses torch.nn for layer definitions.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dl_models.base_model import BaseModel  # Assumes BaseModel implements dynamic layer-loading

class SelfAttentionBlock(nn.Module):
    """
    SelfAttentionBlock computes an attention-weighted representation of features.
    
    Parameters:
      input_dim (int): Dimensionality of the input features.
      attention_dim (int): Dimensionality of the internal attention space.
    """
    def __init__(self, input_dim: int, attention_dim: int):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = attention_dim ** -0.5

    def forward(self, x):
        # x shape: (batch, features)
        q = self.query(x)  # (batch, attention_dim)
        k = self.key(x)    # (batch, attention_dim)
        attn_weights = F.softmax(q * self.scale, dim=-1)  # (batch, attention_dim)
        # Note: For simplicity, compute weighted value per sample.
        weighted = self.value(x) * attn_weights.unsqueeze(-1).squeeze()
        return weighted + x  # Residual connection


class ResidualBlock(nn.Module):
    """
    ResidualBlock implements a simple fully-connected residual module.
    
    Parameters:
      input_dim (int): Input dimension.
      hidden_dim (int): Hidden dimension for the intermediate fully-connected layer.
      dropout_rate (float): Dropout probability.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        # For residual, ensure dimensions match.
        if input_dim != hidden_dim:
            self.residual_mapping = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_mapping = None

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.residual_mapping:
            identity = self.residual_mapping(identity)
        return out + identity


class AdvancedClassifier(BaseModel):
    """
    AdvancedClassifier implements a complex classifier with residual connections and 
    self-attention for dynamic feature aggregation.
    
    Dynamic Layer Configuration includes:
      • Optional multiple ResidualBlock modules.
      • SelfAttentionBlock to re-weight intermediate representations.
      • fc1: Fully-connected layer mapping pooling output to hidden_size.
      • bn1: Batch normalization.
      • activation: ReLU activation.
      • dropout: Dropout to reduce overfitting.
      • fc_out: Final layer mapping hidden_size to num_classes.
    
    Parameters:
      input_size     (int): Dimensionality of the raw input.
      hidden_size    (int): Number of neurons in hidden layers.
      num_classes    (int): Number of output classes.
      attention_dim  (int): Dimension of the self-attention block.
      num_res_blocks (int): Number of ResidualBlock layers to include.
      dropout_rate   (float): Dropout probability.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 attention_dim: int = 32, num_res_blocks: int = 1, dropout_rate: float = 0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout_rate
        super(AdvancedClassifier, self).__init__()
        # Build residual blocks dynamically.
        self.res_blocks = nn.ModuleList([
            ResidualBlock(input_size if i == 0 else hidden_size, hidden_size, dropout_rate)
            for i in range(num_res_blocks)
        ])
        # Self-attention block after residual modules.
        self.attention = SelfAttentionBlock(hidden_size, attention_dim)
        # Final fully-connected layers.
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass for AdvancedClassifier.
        
        Parameters:
          x (Tensor): Input tensor with shape (batch_size, input_size)
        
        Returns:
          Tensor: Logits with shape (batch_size, num_classes)
        """
        out = x
        # Pass through residual blocks.
        for block in self.res_blocks:
            out = block(out)
        # Apply self-attention.
        out = self.attention(out)
        # Fully connected layers.
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        logits = self.fc_out(out)
        return logits


if __name__ == "__main__":
    import torch.optim as optim
    import torch.nn as nn

    # Example usage of the AdvancedClassifier:
    model = AdvancedClassifier(input_size=10, hidden_size=64, num_classes=2, 
                               attention_dim=32, num_res_blocks=2, dropout_rate=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("AdvancedClassifier structure:")
    print(model)

    # Create a dummy input tensor.
    x = torch.randn(5, 10)  # (batch_size, input_size)
    # Forward pass.
    out = model(x)
    print("Output shape:", out.shape)