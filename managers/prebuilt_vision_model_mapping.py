"""
================================================================================
Prebuilt Torchvision Model Mapping Module
--------------------------------------------------------------------------------
This module defines a mapping of prebuilt vision models available from
the torchvision.models package. It provides a standardized interface for
instantiating a wide variety of popular convolutional neural network architectures.

Overview:
  - The module maintains a dictionary, PREBUILT_TORCHVISION_MODEL_MAPPING, where each key
    is a string representing a specific vision model (e.g., "ResNet50", "VGG16").
  - Each dictionary entry has the following keys:
      • "class": A callable for instantiating the model (e.g., models.resnet50).
      • "required_params": A list of parameters required during model instantiation.
      • "optional_params": A dictionary of optional parameters with default values.
        Typically, the "pretrained" flag is provided to decide whether to load
        pre-trained weights.
  - The function create_torchvision_model allows users to create a model instance
    by supplying a model name and a configuration dictionary that can override
    default parameters.

Usage:
  >>> from managers.prebuilt_torchvision_model_mapping import create_torchvision_model
  >>> config = {"pretrained": True}
  >>> model = create_torchvision_model("ResNet50", config)
  >>> print(model)

Dependencies:
  - torch.nn: For PyTorch neural network modules.
  - torchvision.models: Provides callable functions for popular vision models.
================================================================================
"""

import torch.nn as nn
import torchvision.models as models

# Mapping of prebuilt torchvision models.
# Each entry contains:
#    "class": The callable used to instantiate the model.
#    "required_params": List of parameters that must be provided on instantiation.
#    "optional_params": Optional parameters with default values, typically "pretrained".
PREBUILT_TORCHVISION_MODEL_MAPPING = {
    # ----- AlexNet -----
    "AlexNet": {
        "class": models.alexnet,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- VGG Models -----
    "VGG11": {
        "class": models.vgg11,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG11_bn": {
        "class": models.vgg11_bn,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG13": {
        "class": models.vgg13,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG13_bn": {
        "class": models.vgg13_bn,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG16": {
        "class": models.vgg16,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG16_bn": {
        "class": models.vgg16_bn,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG19": {
        "class": models.vgg19,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "VGG19_bn": {
        "class": models.vgg19_bn,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- ResNet Models -----
    "ResNet18": {
        "class": models.resnet18,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ResNet34": {
        "class": models.resnet34,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ResNet50": {
        "class": models.resnet50,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ResNet101": {
        "class": models.resnet101,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ResNet152": {
        "class": models.resnet152,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- SqueezeNet Models -----
    "SqueezeNet1_0": {
        "class": models.squeezenet1_0,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "SqueezeNet1_1": {
        "class": models.squeezenet1_1,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- DenseNet Models -----
    "DenseNet121": {
        "class": models.densenet121,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "DenseNet169": {
        "class": models.densenet169,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "DenseNet161": {
        "class": models.densenet161,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "DenseNet201": {
        "class": models.densenet201,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- Inception and GoogLeNet -----
    "InceptionV3": {
        "class": models.inception_v3,
        "required_params": [],
        "optional_params": {
            "pretrained": False,
            "aux_logits": True,
            "transform_input": False
        }
    },
    "GoogLeNet": {
        "class": models.googlenet,
        "required_params": [],
        "optional_params": {
            "pretrained": False,
            "aux_logits": True,
            "transform_input": False
        }
    },
    # ----- ShuffleNet Models -----
    "ShuffleNetV2_x0_5": {
        "class": models.shufflenet_v2_x0_5,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ShuffleNetV2_x1_0": {
        "class": models.shufflenet_v2_x1_0,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ShuffleNetV2_x1_5": {
        "class": models.shufflenet_v2_x1_5,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "ShuffleNetV2_x2_0": {
        "class": models.shufflenet_v2_x2_0,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- MobileNet Models -----
    "MobileNetV2": {
        "class": models.mobilenet_v2,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "MobileNetV3_large": {
        "class": models.mobilenet_v3_large,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "MobileNetV3_small": {
        "class": models.mobilenet_v3_small,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- EfficientNet Models -----
    "EfficientNetB0": {
        "class": models.efficientnet_b0,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB1": {
        "class": models.efficientnet_b1,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB2": {
        "class": models.efficientnet_b2,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB3": {
        "class": models.efficientnet_b3,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB4": {
        "class": models.efficientnet_b4,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB5": {
        "class": models.efficientnet_b5,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB6": {
        "class": models.efficientnet_b6,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "EfficientNetB7": {
        "class": models.efficientnet_b7,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    # ----- RegNet Models -----
    "RegNetY_400MF": {
        "class": models.regnet_y_400mf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetY_800MF": {
        "class": models.regnet_y_800mf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetY_1_6GF": {
        "class": models.regnet_y_1_6gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetY_3_2GF": {
        "class": models.regnet_y_3_2gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetY_8GF": {
        "class": models.regnet_y_8gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetY_16GF": {
        "class": models.regnet_y_16gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetY_32GF": {
        "class": models.regnet_y_32gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_400MF": {
        "class": models.regnet_x_400mf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_800MF": {
        "class": models.regnet_x_800mf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_1_6GF": {
        "class": models.regnet_x_1_6gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_3_2GF": {
        "class": models.regnet_x_3_2gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_8GF": {
        "class": models.regnet_x_8gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_16GF": {
        "class": models.regnet_x_16gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
    "RegNetX_32GF": {
        "class": models.regnet_x_32gf,
        "required_params": [],
        "optional_params": {"pretrained": False}
    },
}


def create_torchvision_model(model_name: str, config: dict) -> nn.Module:
    """
    Create and return an instance of a prebuilt torchvision model chosen from    
    PREBUILT_TORCHVISION_MODEL_MAPPING.

    Parameters:
      model_name (str): A key from PREBUILT_TORCHVISION_MODEL_MAPPING identifying the
                        desired model.
      config (dict): A dictionary containing model parameters. It must supply values for 
                     keys in "required_params" and can override defaults in "optional_params".

    Returns:
      nn.Module: An instantiated torchvision model.

    Raises:
      ValueError: If model_name is not defined in the mapping or if a required parameter is missing.
    """
    if model_name not in PREBUILT_TORCHVISION_MODEL_MAPPING:
        raise ValueError(
            f"Model {model_name} is not defined in the PREBUILT_TORCHVISION_MODEL_MAPPING.")

    model_info = PREBUILT_TORCHVISION_MODEL_MAPPING[model_name]
    model_class = model_info["class"]
    instance_params = {}

    # Assign required parameters.
    for param in model_info.get("required_params", []):
        if param not in config:
            raise ValueError(
                f"Missing required parameter '{param}' for model '{model_name}'.")
        instance_params[param] = config[param]

    # Process optional parameters: use provided value or default.
    for param, default_val in model_info.get("optional_params", {}).items():
        instance_params[param] = config.get(param, default_val)

    return model_class(**instance_params)


# -----------------------------------------------------------------------------
# Example Usage (for testing purposes)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Create a pretrained ResNet50 model.
    resnet_config = {"pretrained": True}
    resnet_model = create_torchvision_model("ResNet50", resnet_config)
    print("ResNet50 Model:")
    print(resnet_model)

    # Example: Create a pretrained EfficientNetB0 model.
    efficientnet_config = {"pretrained": True}
    efficientnet_model = create_torchvision_model(
        "EfficientNetB0", efficientnet_config)
    print("\nEfficientNetB0 Model:")
    print(efficientnet_model)
