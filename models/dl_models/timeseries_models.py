"""
This module defines several prebuilt sequential models for time series forecasting using
a dynamic layer loading mechanism. Each model inherits from BaseModel (which itself extends
torch.nn.Module) and defines model architecture via a class variable `_layer_configs`.

The dynamic layer loading mechanism allows model layers to be specified as configuration
dictionaries. During initialization, these dictionaries are processed to instantiate
layers either from the basic or the extended layer mappings.

The available models in this file:
  - SimpleLSTMForecast: Uses an LSTM and a Linear layer, taking the last time step's output.
  - SimpleGRUForecast: Uses a GRU and a Linear layer, taking the last time step's output.
  - SimpleTransformerTimeSeries: Uses a Linear layer to project inputs, a Transformer Encoder,
    and a Linear output layer. The Transformer encoder expects inputs in (seq_len, batch_size, feature_dim) format.
  
Each model class includes detailed docstrings explaining the expected parameters and their usage.
"""

import torch.nn as nn
from models.dl_models.base_model import BaseModel

# -----------------------------------------------------------------------------
# Simple LSTM Forecast Model with Dynamic Layer Loading
# -----------------------------------------------------------------------------
class SimpleLSTMForecast(BaseModel):
    """
    SimpleLSTMForecast model for time series forecasting.

    Architecture:
      - An LSTM layer: dynamically configured based on input parameters.
      - A fully-connected (Linear) layer: maps the LSTM's last time step output to the forecast output.

    Dynamic Layer Loading:
      The layer configuration is defined in the `_layer_configs` class variable.
      Each dictionary in this list specifies:
         "name": attribute name (e.g., 'lstm', 'fc')
         "layer_name": key used to look up the layer from the basic or extended layer maps.
         "config": parameters required for the layer's initialization. These can be either fixed values
                   or callables (e.g., lambda self: self.input_size) which are evaluated during initialization.

    Required Parameters for Initialization:
      - input_size (int): Dimensionality of the input features.
      - hidden_size (int): Number of features in the hidden state.
      - num_layers (int): Number of recurrent layers.
      - output_size (int): Dimensionality of the forecast output.
      - dropout (float, optional): Dropout probability in the LSTM (default: 0.0).
    """
    _layer_configs = [
        {
            "name": "lstm",
            "layer_name": "LSTM",
            "config": {
                "input_size": lambda self: self.input_size,
                "hidden_size": lambda self: self.hidden_size,
                "num_layers": lambda self: self.num_layers,
                "batch_first": True,  # Fixed value because input data is expected in (batch, seq, feature)
                "dropout": lambda self: self.dropout
            }
        },
        {
            "name": "fc",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.hidden_size,
                "out_features": lambda self: self.output_size
            }
        }
    ]

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        """
        Initialize the SimpleLSTMForecast model.

        Parameters:
          input_size (int): Number of input features per time step
          hidden_size (int): Hidden state size for LSTM
          num_layers (int): Number of LSTM layers
          output_size (int): Number of output features
          dropout (float): Dropout probability in LSTM layers (default 0.0)
        """
        # Store initialization parameters that may be required dynamically.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        # Invoke dynamic layer creation in the BaseModel initializer.
        super(SimpleLSTMForecast, self).__init__()

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
          x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
          Tensor: Forecast output of shape (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)  # LSTM output shape: (batch_size, seq_len, hidden_size)
        out = self.fc(lstm_out[:, -1, :])  # Use the output from the last time step
        return out


# -----------------------------------------------------------------------------
# Simple GRU Forecast Model with Dynamic Layer Loading
# -----------------------------------------------------------------------------
class SimpleGRUForecast(BaseModel):
    """
    SimpleGRUForecast model for time series forecasting.

    Architecture:
      - A GRU layer: dynamically configured based on provided parameters.
      - A fully-connected (Linear) layer: maps the last GRU output to the final output.

    Dynamic Layer Loading:
      Defined similarly to SimpleLSTMForecast using the `_layer_configs` class variable.

    Required Parameters for Initialization are the same as SimpleLSTMForecast.
    """
    _layer_configs = [
        {
            "name": "gru",
            "layer_name": "GRU",
            "config": {
                "input_size": lambda self: self.input_size,
                "hidden_size": lambda self: self.hidden_size,
                "num_layers": lambda self: self.num_layers,
                "batch_first": True,  # Expect input in (batch, seq, feature) format
                "dropout": lambda self: self.dropout
            }
        },
        {
            "name": "fc",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.hidden_size,
                "out_features": lambda self: self.output_size
            }
        }
    ]

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        """
        Initialize the SimpleGRUForecast model.

        Parameters:
          input_size (int): Number of input features per time step
          hidden_size (int): Hidden state dimensionality for GRU
          num_layers (int): Number of GRU layers
          output_size (int): Number of output features
          dropout (float): Dropout probability (default 0.0)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        super(SimpleGRUForecast, self).__init__()

    def forward(self, x):
        """
        Forward pass through the GRU model.

        Parameters:
          x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
          Tensor: Model output of shape (batch_size, output_size)
        """
        gru_out, _ = self.gru(x)  # GRU output shape: (batch_size, seq_len, hidden_size)
        out = self.fc(gru_out[:, -1, :])  # Use the output of the last time step
        return out


# -----------------------------------------------------------------------------
# Simple Transformer for Time Series Forecasting with Dynamic Layer Loading
# -----------------------------------------------------------------------------
class SimpleTransformerTimeSeries(BaseModel):
    """
    SimpleTransformerTimeSeries model for time series forecasting using a transformer-based encoder.

    Architecture:
      - Input Linear layer: Projects input features to a higher-dimensional space (hidden_dim).
      - Transformer Encoder: Processes the sequence using self-attention mechanisms.
      - Output Linear layer (fc_out): Maps the final encoder output (last time step)
                                     to the forecast output dimensions.

    Dynamic Layer Loading:
      The model layers are defined in the `_layer_configs` class variable.
      Note that for the transformer encoder, the encoder layer is created dynamically (via a lambda)
      using nn.TransformerEncoderLayer, and then passed to nn.TransformerEncoder.

    Required Parameters for Initialization:
      - input_size (int): Dimensionality of input features per time step.
      - hidden_dim (int): Output dimensionality of the input linear layer and model encoder (d_model).
      - num_heads (int): Number of attention heads in the transformer encoder.
      - num_encoder_layers (int): Number of transformer encoder layers.
      - output_size (int): Dimensionality of the final output.
      - dropout (float, optional): Dropout probability in transformer encoder layers (default: 0.1).
    """
    _layer_configs = [
        {
            "name": "input_linear",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.input_size,
                "out_features": lambda self: self.hidden_dim
            }
        },
        {
            "name": "transformer_encoder",
            "layer_name": "TransformerEncoder",
            "config": {
                # The encoder_layer is dynamically created using a lambda function.
                "encoder_layer": lambda self: nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout
                ),
                "num_layers": lambda self: self.num_encoder_layers,
                "norm": None
            }
        },
        {
            "name": "fc_out",
            "layer_name": "Linear",
            "config": {
                "in_features": lambda self: self.hidden_dim,
                "out_features": lambda self: self.output_size
            }
        }
    ]

    def __init__(self, input_size: int, hidden_dim: int, num_heads: int, num_encoder_layers: int, output_size: int, dropout: float = 0.1):
        """
        Initialize the SimpleTransformerTimeSeries model.

        Parameters:
          input_size (int): Number of input features per time step.
          hidden_dim (int): Dimensionality of the transformer encoder (d_model).
          num_heads (int): Number of attention heads in the transformer.
          num_encoder_layers (int): Number of transformer encoder layers.
          output_size (int): Number of output features.
          dropout (float): Dropout probability for encoder layers (default 0.1).
        """
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.output_size = output_size
        self.dropout = dropout
        super(SimpleTransformerTimeSeries, self).__init__()

    def forward(self, x):
        """
        Forward pass through the Transformer model for time series data.

        Parameters:
          x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
          Tensor: Forecast output of shape (batch_size, output_size)
        """
        x = self.input_linear(x)
        # The transformer encoder expects input of shape (seq_len, batch_size, feature_dim)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        # Use the output from the last time step of the encoder.
        out = self.fc_out(transformer_out[-1])
        return out
