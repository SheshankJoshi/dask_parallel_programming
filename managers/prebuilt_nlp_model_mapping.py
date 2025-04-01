"""
================================================================================
Prebuilt NLP Model Mapping Module
--------------------------------------------------------------------------------
This module provides a standardized interface for instantiating prebuilt 
transformer-based NLP models, primarily from the Hugging Face Transformers library.

Overview:
  - The module defines a mapping dictionary, PREBUILT_NLP_MODEL_MAPPING, in which
    each key is a string identifier for a model (e.g., "BertModel", "GPT2Model"),
    and the value is a configuration dictionary.
  - Each configuration dictionary contains:
      • "class": The actual model class (callable) from the transformers library.
      • "required_params": A (usually empty) list of parameters needed.
      • "optional_params": A dict of optional parameters with default values.
        Common keys include "pretrained", specifying either a model identifier
        (to load pretrained weights) or False (to instantiate the model from scratch),
        and "config", which can be used to pass a configuration object.
  - The create_nlp_model function instantiates and returns a model based on the 
    identifier and a provided config dictionary. If a pretrained value is provided,
    the model’s from_pretrained method is used.

Usage:
  >>> from managers.prebuilt_nlp_model_mapping import create_nlp_model
  >>> config = {"pretrained": "bert-base-uncased"}
  >>> model = create_nlp_model("BertModel", config)
  >>> print(model)

Dependencies:
  - torch.nn: For compatibility with PyTorch, as all models are subclasses of nn.Module.
  - transformers: Provides the actual model implementations.
================================================================================
"""

import torch.nn as nn

# Import Hugging Face transformer models - extended list.
from transformers import (
    BertModel,
    DistilBertModel,
    RobertaModel,
    XLNetModel,
    GPT2Model,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    ElectraModel,
    AlbertModel,
    DebertaModel,    # Depending on your transformers version, it could be DebertaV2Model.
    LongformerModel,
    CamembertModel,
    PegasusForConditionalGeneration,
)

# Mapping of prebuilt NLP models for encoder, decoder, and encoder-decoder architectures.
# Each dictionary entry contains:
#   - "class": The callable (model class) to instantiate the model.
#   - "required_params": A list of parameters that must be provided (typically empty).
#   - "optional_params": A dictionary of optional parameters with default values.
#     Key 'pretrained': a string identifier for loading pretrained weights, or False.
#     Key 'config': additional model configuration if needed.
PREBUILT_NLP_MODEL_MAPPING = {
    # ----- Encoder Models -----
    "BertModel": {
        "class": BertModel,
        "required_params": [],
        "optional_params": {"pretrained": "bert-base-uncased", "config": None},
    },
    "DistilBertModel": {
        "class": DistilBertModel,
        "required_params": [],
        "optional_params": {"pretrained": "distilbert-base-uncased", "config": None},
    },
    "RobertaModel": {
        "class": RobertaModel,
        "required_params": [],
        "optional_params": {"pretrained": "roberta-base", "config": None},
    },
    "XLNetModel": {
        "class": XLNetModel,
        "required_params": [],
        "optional_params": {"pretrained": "xlnet-base-cased", "config": None},
    },
    "ElectraModel": {
        "class": ElectraModel,
        "required_params": [],
        "optional_params": {"pretrained": "google/electra-small-discriminator", "config": None},
    },
    "AlbertModel": {
        "class": AlbertModel,
        "required_params": [],
        "optional_params": {"pretrained": "albert-base-v2", "config": None},
    },
    "DebertaModel": {
        "class": DebertaModel,
        "required_params": [],
        "optional_params": {"pretrained": "microsoft/deberta-base", "config": None},
    },
    "LongformerModel": {
        "class": LongformerModel,
        "required_params": [],
        "optional_params": {"pretrained": "allenai/longformer-base-4096", "config": None},
    },
    "CamembertModel": {
        "class": CamembertModel,
        "required_params": [],
        "optional_params": {"pretrained": "camembert-base", "config": None},
    },
    # ----- Decoder / Causal Models -----
    "GPT2Model": {
        "class": GPT2Model,
        "required_params": [],
        "optional_params": {"pretrained": "gpt2", "config": None},
    },
    # ----- Encoder-Decoder Models -----
    "T5ForConditionalGeneration": {
        "class": T5ForConditionalGeneration,
        "required_params": [],
        "optional_params": {"pretrained": "t5-small", "config": None},
    },
    "BartForConditionalGeneration": {
        "class": BartForConditionalGeneration,
        "required_params": [],
        "optional_params": {"pretrained": "facebook/bart-large", "config": None},
    },
    "PegasusForConditionalGeneration": {
        "class": PegasusForConditionalGeneration,
        "required_params": [],
        "optional_params": {"pretrained": "google/pegasus-xsum", "config": None},
    },
}

def create_nlp_model(model_name: str, config: dict) -> nn.Module:
    """
    Create and return an instance of a prebuilt NLP model.

    For transformer models, if the "pretrained" parameter is specified and not False,
    the model is loaded via its `from_pretrained` method.

    Parameters:
      model_name (str): A key from PREBUILT_NLP_MODEL_MAPPING identifying the model.
      config (dict): A dictionary of parameters to initialize the model.
          It should contain:
            - All items listed in "required_params".
            - Overrides for those in "optional_params" if desired.

    Returns:
      nn.Module: An instantiated NLP model.

    Raises:
      ValueError: If model_name is not found in PREBUILT_NLP_MODEL_MAPPING or if a
                  required parameter is missing.
    """
    if model_name not in PREBUILT_NLP_MODEL_MAPPING:
        raise ValueError(f"Model {model_name} is not defined in the PREBUILT_NLP_MODEL_MAPPING.")
    
    model_info = PREBUILT_NLP_MODEL_MAPPING[model_name]
    model_class = model_info["class"]
    instance_params = {}

    # Ensure required parameters are provided.
    for param in model_info.get("required_params", []):
        if param not in config:
            raise ValueError(f"Missing required parameter '{param}' for model '{model_name}'.")
        instance_params[param] = config[param]

    # Process optional parameters with overrides.
    for param, default_val in model_info.get("optional_params", {}).items():
        instance_params[param] = config.get(param, default_val)

    # Extract values for loading pretrained weights.
    pretrained_value = instance_params.pop("pretrained", False)
    model_config = instance_params.pop("config", None)

    if pretrained_value:
        # If a configuration is provided, use it.
        if model_config is not None:
            return model_class.from_pretrained(pretrained_value, config=model_config, **instance_params)
        else:
            return model_class.from_pretrained(pretrained_value, **instance_params)
    else:
        # Direct instantiation of the model.
        return model_class(**instance_params)

# -----------------------------------------------------------------------------
# Example Usage (for testing purposes)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Create a pretrained BERT model.
    bert_config = {"pretrained": "bert-base-uncased"}
    bert_model = create_nlp_model("BertModel", bert_config)
    print("BertModel:")
    print(bert_model)

    # Example: Create a GPT2 model from scratch.
    gpt2_config = {"pretrained": False}
    gpt2_model = create_nlp_model("GPT2Model", gpt2_config)
    print("\nGPT2Model:")
    print(gpt2_model)

    # Example: Create a T5 encoder-decoder model using pretrained weights.
    t5_config = {"pretrained": "t5-small"}
    t5_model = create_nlp_model("T5ForConditionalGeneration", t5_config)
    print("\nT5ForConditionalGeneration:")
    print(t5_model)