"""
================================================================================
Hugging Face Model Loader Module
--------------------------------------------------------------------------------
This module provides a comprehensive interface for loading Hugging Face models 
for inference or further fine-tuning. It includes:

  - Environment and dependency checks.
  - Authentication mechanism using Hugging Face access tokens.
  - Methods to load models and tokenizers based on task (e.g. sequence classification,
    text generation, feature extraction, etc.).
  - Detailed logging and error handling to assist with troubleshooting.

Usage Example:
    >>> from managers.hugging_face_model_loader import HuggingFaceModelLoader
    >>> # Authenticate using your Hugging Face access token.
    >>> loader = HuggingFaceModelLoader()
    >>> loader.authenticate(token="your_hf_token")
    >>> # Load a model and tokenizer for sequence classification.
    >>> model, tokenizer = loader.load_model(
    ...     model_name="distilbert-base-uncased-finetuned-sst-2-english", 
    ...     task="sequence-classification"
    ... )
    >>> # Use the loaded model and tokenizer for inference.
    >>> inputs = tokenizer("The movie was fantastic!", return_tensors="pt")
    >>> outputs = model(**inputs)
    >>> print(outputs)

Dependencies:
    - transformers: For model and tokenizer classes.
    - huggingface_hub: For authentication and API interactions.
    - torch: For tensor operations (if using PyTorch models).
    
Install dependencies via:
    pip install transformers huggingface_hub torch
================================================================================
"""

import os
import logging
from typing import Tuple, Optional

try:
    # Import required modules from transformers and huggingface_hub
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoFeatureExtractor,
        pipeline,
    )
except ImportError as e:
    raise ImportError("Please install transformers using: pip install transformers") from e

try:
    from huggingface_hub import login, HfApi
except ImportError as e:
    raise ImportError("Please install huggingface_hub using: pip install huggingface_hub") from e

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceModelLoader:
    """
    HuggingFaceModelLoader provides methods to authenticate with Hugging Face,
    load models and tokenizers suitable for various tasks, and perform inference 
    or further fine-tuning.
    
    Attributes:
        hf_token (Optional[str]): The Hugging Face access token for authentication.
    
    Methods:
        authenticate(token: str) -> None:
            Authenticate with the Hugging Face hub using the provided token.
            
        load_model(model_name: str, task: str) -> Tuple[object, Optional[object]]:
            Load a model (and tokenizer if applicable) for the given task.
            
        list_models(filter: str = "") -> list:
            List models available on Hugging Face Hub matching a filter.
    """
    def __init__(self):
        self.hf_token: Optional[str] = None
        self.api = HfApi()

    def authenticate(self, token: str) -> None:
        """
        Authenticate with the Hugging Face hub using the given token.
        
        Parameters:
            token (str): The Hugging Face access token.
            
        Raises:
            ValueError: If token is not provided.
        """
        if not token:
            raise ValueError("A valid Hugging Face token must be provided for authentication.")
        self.hf_token = token
        login(token=token)
        logger.info("Authentication successful with Hugging Face.")

    def load_model(self, model_name: str, task: str) -> Tuple[object, Optional[object]]:
        """
        Load a model (and its associated tokenizer or feature extractor) for the specified task.
        
        Parameters:
            model_name (str): The name or identifier of the model on Hugging Face Hub.
            task (str): The task for which the model is intended. Supported tasks include:
                - "sequence-classification"
                - "text-generation"
                - "feature-extraction"
                - "causal-lm"
                - "token-classification"
                - "question-answering"
                - "image-classification"
                - "object-detection"
                (Additional tasks can be added as needed.)
        
        Returns:
            Tuple[object, Optional[object]]:
                A tuple containing the loaded model and the tokenizer/feature extractor if applicable.
                For tasks that do not require a tokenizer/feature extractor, the second element will be None.
        
        Raises:
            ValueError: If the task is unsupported.
            Exception: If there is an error during model/tokenizer loading.
        """
        logger.info(f"Attempting to load model '{model_name}' for task '{task}'")
        model = None
        processor = None  # Can be tokenizer or feature extractor
        
        try:
            if task.lower() == "sequence-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
                processor = AutoTokenizer.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            elif task.lower() in ["text-generation", "causal-lm"]:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
                processor = AutoTokenizer.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            elif task.lower() == "feature-extraction":
                model = AutoModel.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            elif task.lower() == "token-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
                processor = AutoTokenizer.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            elif task.lower() == "question-answering":
                # For QA, usually a pipeline is used directly, but we can load model and tokenizer.
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
                processor = AutoTokenizer.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            elif task.lower() == "image-classification":
                # For image tasks, a feature extractor is often used.
                model = AutoModel.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            else:
                raise ValueError(f"Task '{task}' is not supported. Supported tasks include: "
                                 "'sequence-classification', 'text-generation', 'feature-extraction', "
                                 "'causal-lm', 'token-classification', 'question-answering', 'image-classification'.")
        except Exception as e:
            logger.error(f"Error loading model '{model_name}' for task '{task}': {e}")
            raise Exception(f"Failed to load model/tokenizer for '{model_name}'.") from e

        logger.info(f"Model '{model_name}' loaded successfully.")
        return model, processor

    def list_models(self, filter: str = "") -> list:
        """
        List models available on Hugging Face Hub that match a given filter string.
        
        Parameters:
            filter (str): A keyword to filter the models (default is an empty string which returns all).
        
        Returns:
            list: A list of model identifiers available on the hub.
        """
        try:
            models = self.api.list_models(filter=filter, use_auth_token=self.hf_token)
            model_names = [model.modelId for model in models]
            logger.info(f"Found {len(model_names)} models matching filter '{filter}'.")
            return model_names
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise Exception("Failed to list models from Hugging Face Hub.") from e


# -----------------------------------------------------------------------------
# Example Usage:
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create an instance of the loader.
    loader = HuggingFaceModelLoader()

    # Authenticate with your Hugging Face token.
    # Replace 'your_hf_token' with your actual Hugging Face access token.
    try:
        token = os.environ.get("HF_TOKEN", "")  # Optionally, set your token as an environment variable.
        if not token:
            raise ValueError("Hugging Face token not provided. Set HF_TOKEN environment variable or pass it directly.")
        loader.authenticate(token=token)
    except Exception as auth_error:
        logger.error(f"Authentication failed: {auth_error}")
        exit(1)

    # Load a model and its tokenizer for sequence classification.
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sequence-classification"
        model, tokenizer = loader.load_model(model_name=model_name, task=task)

        # Perform a quick inference test.
        text = "The movie was absolutely fantastic!"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logger.info("Inference successful. Model outputs:")
        logger.info(outputs)
    except Exception as load_error:
        logger.error(f"Model loading failed: {load_error}")