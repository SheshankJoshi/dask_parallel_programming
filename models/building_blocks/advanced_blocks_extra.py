"""
================================================================================
Advanced Blocks Extra (Dynamic _layer_configs Approach)
--------------------------------------------------------------------------------
This module defines several generic building blocks using the dynamic layer-
configuration (_layer_configs) methodology, similar to SimpleClassifier. These 
blocks can be integrated into various architectures beyond classification.

Included Blocks:
  • ConvolutionalBlock:
      A basic convolutional unit for vision tasks, comprising:
        - 2D Convolution
        - Batch Normalization
        - ReLU activation
        - Optional MaxPooling (conditional layer)
  • TransformerEncoderBlock:
      A single transformer encoder layer built with:
        - Multihead Self-Attention
        - Feedforward network (Linear + ReLU + Dropout + Linear)
        - Residual connections and Layer Normalization
  • RecurrentBlock:
      A recurrent neural network block based on LSTM for sequence modeling.
      (Assumes that LSTM is available in your layer mapping if desired.)
      
Usage Example:
    >>> from models.dl_models.advanced_blocks_extra_dynamic import (
    ...     ConvolutionalBlock, TransformerEncoderBlock, RecurrentBlock
    ... )
    >>> import torch
    >>> # Test ConvolutionalBlock.
    >>> conv_block = ConvolutionalBlock(
    ...     in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1,
    ...     use_pooling=True, pool_kernel=2)
    >>> x_conv = torch.randn(8, 3, 32, 32)
    >>> print("ConvolutionalBlock output shape:", conv_block(x_conv).shape)
    >>>
    >>> # Test TransformerEncoderBlock.
    >>> transformer_block = TransformerEncoderBlock(d_model=64, nhead=8, dim_feedforward=128, dropout=0.1)
    >>> x_transformer = torch.randn(10, 8, 64)  # (sequence length, batch, d_model)
    >>> print("TransformerEncoderBlock output shape:", transformer_block(x_transformer).shape)
    >>>
    >>> # Test RecurrentBlock.
    >>> recurrent_block = RecurrentBlock(input_size=50, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True)
    >>> x_recurrent = torch.randn(8, 20, 50)  # (batch, seq_len, input_size)
    >>> print("RecurrentBlock output shape:", recurrent_block(x_recurrent).shape)

Dependencies:
  - torch and torch.nn for defining layers.
  - models/dl_models/base_model.py must implement dynamic _layer_configs.
  - Ensure your layer mapping modules (e.g., layer_map_basic, layer_map_extended)
    contain the keys "Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", 
    "MultiheadAttention", "Linear", "LayerNorm", "Dropout", and optionally "LSTM".
================================================================================
"""

import torch
import torch.nn as nn
from models.dl_models.base_model import BaseModel  # BaseModel must create layers from _layer_configs

###############################################################################
# ConvolutionalBlock using _layer_configs
###############################################################################
class ConvolutionalBlock(BaseModel):
    """
    ConvolutionalBlock implements a convolutional unit for vision tasks.
    It dynamically instantiates its layers:
      • conv: A Conv2d layer.
      • bn: BatchNorm2d.
      • activation: ReLU.
      • pool (conditional): MaxPool2d if use_pooling is True.
      
    Parameters:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      kernel_size (int or tuple): Convolution kernel size.
      stride (int or tuple): Convolution stride.
      padding (int or tuple): Zero-padding applied to input.
      use_pooling (bool): Enables MaxPool2d if True.
      pool_kernel (int or tuple): Kernel size for MaxPool2d.
    """
    _layer_configs = [
        {
            "name": "conv",
            "layer_name": "Conv2d",
            "config": {
                "in_channels": lambda self: self.in_channels,
                "out_channels": lambda self: self.out_channels,
                "kernel_size": lambda self: self.kernel_size,
                "stride": lambda self: self.stride,
                "padding": lambda self: self.padding,
            }
        },
        {
            "name": "bn",
            "layer_name": "BatchNorm2d",
            "config": {
                "num_features": lambda self: self.out_channels
            }
        },
        {
            "name": "activation",
            "layer_name": "ReLU",
            "config": {
                "inplace": True
            }
        },
        {
            "name": "pool",
            "layer_name": "MaxPool2d",
            "config": {
                "kernel_size": lambda self: self.pool_kernel
            },
            "conditional": lambda self: self.use_pooling
        }
    ]
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 use_pooling: bool = False, pool_kernel=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_pooling = use_pooling
        self.pool_kernel = pool_kernel
        super(ConvolutionalBlock, self).__init__()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.use_pooling:
            x = self.pool(x)
        return x

###############################################################################
# TransformerEncoderBlock using _layer_configs
###############################################################################
class TransformerEncoderBlock(BaseModel):
    """
    TransformerEncoderBlock implements a transformer encoder layer using dynamic
    layer configuration. The block includes:
      • self_attn: MultiheadAttention.
      • linear1: Linear layer for feedforward (expansion).
      • activation: ReLU activation.
      • dropout: Dropout following activation.
      • linear2: Linear layer to project back to d_model.
      • norm1: LayerNorm after attention.
      • norm2: LayerNorm after feedforward.
      
    Parameters:
      d_model (int): Dimension of the input embedding.
      nhead (int): Number of attention heads.
      dim_feedforward (int): Dimension of the feedforward network.
      dropout (float): Dropout probability.
    """
    _layer_configs = [
        {
            "name": "self_attn",
            "layer_name": "MultiheadAttention",
            "config": {
                "embed_dim": lambda self: self.d_model,
                "num_heads": lambda self: self.nhead,
                "dropout": lambda self: self.dropout
            }
        },
        {
            "name": "linear1",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.d_model,
                "out_features": lambda self: self.dim_feedforward
            }
        },
        {
            "name": "activation",
            "layer_name": "ReLU",
            "config": {
                "inplace": True
            }
        },
        {
            "name": "dropout_layer",
            "layer_name": "Dropout",
            "config": {
                "p": lambda self: self.dropout
            }
        },
        {
            "name": "linear2",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.dim_feedforward,
                "out_features": lambda self: self.d_model
            }
        },
        {
            "name": "norm1",
            "layer_name": "LayerNorm",
            "config": {
                "normalized_shape": lambda self: self.d_model
            }
        },
        {
            "name": "norm2",
            "layer_name": "LayerNorm",
            "config": {
                "normalized_shape": lambda self: self.d_model
            }
        }
    ]
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        super(TransformerEncoderBlock, self).__init__()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention expects input of shape (sequence length, batch, d_model)
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src = src + attn_output  # Residual connection
        src = self.norm1(src)
        
        ff_output = self.linear2(self.dropout_layer(self.activation(self.linear1(src))))
        src = src + ff_output  # Residual connection
        src = self.norm2(src)
        return src

###############################################################################
# RecurrentBlock using _layer_configs
###############################################################################
class RecurrentBlock(BaseModel):
    """
    RecurrentBlock implements an LSTM-based recurrent unit for sequence modeling.
    It constructs the LSTM dynamically via _layer_configs.
    
    Parameters:
      input_size (int): Dimensionality of the input features.
      hidden_size (int): Number of features in the hidden state.
      num_layers (int): Number of stacked LSTM layers.
      dropout (float): Dropout probability (between LSTM layers).
      bidirectional (bool): Use a bidirectional LSTM if True.
    """
    _layer_configs = [
        {
            "name": "lstm",
            "layer_name": "LSTM",
            "config": {
                "input_size": lambda self: self.input_size,
                "hidden_size": lambda self: self.hidden_size,
                "num_layers": lambda self: self.num_layers,
                "dropout": lambda self: self.dropout,
                "bidirectional": lambda self: self.bidirectional,
                # batch_first is typically fixed.
                "batch_first": lambda self: True
            }
        }
    ]
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        super(RecurrentBlock, self).__init__()
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        out, _ = self.lstm(x)
        return out

###############################################################################
# Testing the blocks
###############################################################################
if __name__ == "__main__":
    import torch

    # Test ConvolutionalBlock.
    conv_block = ConvolutionalBlock(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1,
        use_pooling=True, pool_kernel=2)
    x_conv = torch.randn(8, 3, 32, 32)
    conv_out = conv_block(x_conv)
    print("ConvolutionalBlock output shape:", conv_out.shape)  # Expected: (8, 16, 16, 16)

    # Test TransformerEncoderBlock.
    transformer_block = TransformerEncoderBlock(d_model=64, nhead=8, dim_feedforward=128, dropout=0.1)
    x_transformer = torch.randn(10, 8, 64)  # (seq_len, batch, d_model)
    transformer_out = transformer_block(x_transformer)
    print("TransformerEncoderBlock output shape:", transformer_out.shape)  # Expected: (10, 8, 64)

    # Test RecurrentBlock.
    recurrent_block = RecurrentBlock(input_size=50, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True)
    x_recurrent = torch.randn(8, 20, 50)  # (batch, seq_len, input_size)
    recurrent_out = recurrent_block(x_recurrent)
    # If bidirectional, output hidden dim becomes 2 * hidden_size
    print("RecurrentBlock output shape:", recurrent_out.shape)