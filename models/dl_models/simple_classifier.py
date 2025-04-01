"""
================================================================================
SimpleClassifier Module
--------------------------------------------------------------------------------
This module defines a simple fully-connected neural network classifier using a 
dynamic layer-loading mechanism inherited from BaseModel. The SimpleClassifier 
demonstrates how to declare layers as a class variable (_layer_configs) that 
specify the architecture via configuration dictionaries.

Architecture:
  - fc1: A fully-connected (Linear) layer that projects input features (input_size)
         to a hidden representation (hidden_size).
  - activation: A ReLU activation function applied in-place.
  - fc2: A fully-connected (Linear) layer that maps from the hidden representation 
         (hidden_size) to the number of classes (num_classes).

Dynamic Layer Loading:
  - By inheriting from BaseModel, the SimpleClassifier automatically instantiates 
    layers defined in _layer_configs. Each configuration includes:
       • "name": The attribute name assigned for the layer (e.g., "fc1").
       • "layer_name": The key used to look up the layer from the layer mappings. 
                       For example, "Linear" or "ReLU".
       • "config": A dict of initialization parameters where values can be either 
                    fixed values or callables (lambdas) that reference instance attributes.
                    
Usage Example:
  >>> model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=2)
  >>> print(model)
  >>> x = torch.randn(batch_size, input_size)
  >>> out = model(x)
  >>> print("Output shape:", out.shape)

Dependencies:
  - The module relies on models.dl_models.base_model.BaseModel for dynamic 
    layer instantiation. It also uses torch.nn for layers and torch.optim 
    for training.
================================================================================
"""

from models.dl_models.base_model import BaseModel

class SimpleClassifier(BaseModel):
    """
    SimpleClassifier is a straightforward fully-connected classifier utilizing 
    dynamic layer configuration provided by BaseModel.

    Attributes (via dynamic layer loading):
      fc1         -- First Linear layer; input features mapped to hidden_size.
      activation  -- ReLU activation (in-place).
      fc2         -- Second Linear layer; hidden_size mapped to num_classes.

    Parameters for Initialization:
      input_size  (int): Number of features in the input.
      hidden_size (int): Dimensionality of the hidden layer.
      num_classes (int): Number of output classes for classification.
    """
    # _layer_configs defines the model's architecture using a list of configuration dictionaries.
    _layer_configs = [
        {
            "name": "fc1",
            "layer_name": "Linear",
            "config": {
                # in_features is dynamically obtained from self.input_size
                "in_features": lambda self: self.input_size,
                # out_features is dynamically obtained from self.hidden_size
                "out_features": lambda self: self.hidden_size
            }
        },
        {
            "name": "activation",
            "layer_name": "ReLU",
            "config": {
                # Using inplace ReLU activation for efficiency.
                "inplace": True
            }
        },
        {
            "name": "fc2",
            "layer_name": "Linear",
            "config": {
                # in_features is dynamically obtained from self.hidden_size
                "in_features": lambda self: self.hidden_size,
                # out_features is dynamically obtained from self.num_classes
                "out_features": lambda self: self.num_classes
            }
        }
    ]
  
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        Initialize the SimpleClassifier by storing configuration parameters 
        and delegating layer instantiation to BaseModel.

        Parameters:
          input_size  (int): Dimensionality of the input features.
          hidden_size (int): Number of features for the hidden layer.
          num_classes (int): Number of output classes.
        """
        # Storing initialization parameters for reference in layer-config lambdas.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # Call the BaseModel initializer to instantiate and set up layers.
        super(SimpleClassifier, self).__init__()

    def forward(self, x):
        """
        Defines the forward pass of the model. Input x is propagated through the layers
        in the order defined in _layer_configs.

        Parameters:
          x (Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
          Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Sequentially apply the dynamically created layers.
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    import torch
    import torch.optim as optim
    import torch.nn as nn

    # Example usage:
    # Create an instance of SimpleClassifier with specified parameters.
    model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=2)
    # Set up the optimizer and loss criterion.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Print the model structure.
    print(model)

    # Create a dummy input batch.
    x = torch.randn(5, 10)  # (batch_size, input_size)
    # Compute the output of the classifier.
    out = model(x)
    print("Output shape:", out.shape)