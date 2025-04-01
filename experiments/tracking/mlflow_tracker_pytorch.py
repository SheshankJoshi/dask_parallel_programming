"""
================================================================================
MLflow Tracker for PyTorch Experiments
--------------------------------------------------------------------------------
This module provides a comprehensive framework for tracking PyTorch experiments 
using MLflow. It is designed to:

  - Create or retrieve an MLflow experiment by name.
  - Set up a training loop for a PyTorch model on provided data.
  - Log hyperparameters, training metrics (train and validation loss) across epochs.
  - Output run details including experiment ID, run ID, and final metrics.

This module is intended to be used in experimental workflows where tracking 
of model performance, hyperparameter configuration, and training progress is crucial.

Modules and Functions:
  • create_or_get_experiment(experiment_name: str) -> str
       - Checks if an experiment exists in MLflow by name; if not, creates a new one.
       - Returns the experiment ID.

  • run_pytorch_experiment_mlflow(
           model: torch.nn.Module,
           dataloader: torch.utils.data.DataLoader,
           criterion: torch.nn.Module,
           optimizer: torch.optim.Optimizer,
           hidden_size: int,
           experiment_name: str = "pytorch_experiment", 
           epochs: int = 20,
           learning_rate: float = 0.001,
       ) -> dict
       - Runs a simple training loop for the provided model using the given DataLoader,
         loss function, optimizer, and hyperparameters.
       - Logs parameters and metrics to MLflow, including training and simulated
         validation loss for each epoch.
       - Returns a dictionary containing run details such as experiment ID, run ID,
         final training loss, and final validation loss.

Usage Example:
    See the __main__ block at the end of the module for an end-to-end example
    including dummy data generation, DataLoader creation, model instantiation, and 
    running the experiment with MLflow tracking.

Dependencies:
    - mlflow: For experiment tracking.
    - torch, numpy: For model training and numerical computations.
    - time, random: To simulate epoch duration and validation loss noise.
================================================================================
"""

import mlflow
import time
import random
import numpy as np
import torch
from typing import Type

def create_or_get_experiment(experiment_name: str) -> str:
    """
    Create a new MLflow experiment if it does not already exist, or return the 
    existing experiment ID.

    Parameters:
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        str: The experiment ID corresponding to the given experiment name.
    
    Behavior:
        - If an experiment with the given name is found in MLflow, its experiment ID
          is returned.
        - Otherwise, a new experiment is created with that name and its experiment ID 
          is returned.
    
    Example:
        >>> experiment_id = create_or_get_experiment("my_experiment")
        >>> print("Experiment ID:", experiment_id)
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment with id: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Found existing experiment with id: {experiment_id}")
    return experiment_id


def run_pytorch_experiment_mlflow(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    hidden_size: int,
    experiment_name: str = "pytorch_experiment", 
    epochs: int = 20,
    learning_rate: float = 0.001,
) -> dict:
    """
    Run a PyTorch experiment with MLflow tracking.

    This function sets up a complete training loop with the following capabilities:
      - Reproducibility: Sets manual seeds for torch and numpy.
      - Experiment Management: Uses MLflow to create or get an experiment by name.
      - Hyperparameter Logging: Logs hyperparameters such as number of epochs, learning 
        rate, hidden layer size, batch size, input size, and number of classes.
      - Training Loop: Iterates over the given number of epochs, training the model 
        on each batch:
          • Zeroes gradients.
          • Computes the model output, loss, and performs backpropagation.
          • Updates model parameters using the optimizer.
      - Metrics Logging: Logs:
          • Average training loss per epoch.
          • A simulated validation loss per epoch (for demonstration).
          • Metrics are logged using MLflow's logging functionality.
      - Run Information: On completion, returns a dictionary including the experiment ID,
        run ID, final training loss, and final validation loss.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data batches.
        criterion (torch.nn.Module): Loss function to measure model performance.
        optimizer_class (torch.optim.Optimizer): Optimizer class for training (e.g., torch.optim.Adam).
        hidden_size (int): The hyperparameter value for hidden layer size used in the model.
        experiment_name (str): Name to use for the MLflow experiment (default "pytorch_experiment").
        epochs (int): Number of training epochs (default 20).
        learning_rate (float): Learning rate for training (default 0.001).

    Returns:
        dict: A dictionary containing:
            - "experiment_id": ID of the MLflow experiment.
            - "run_id": ID of the MLflow run.
            - "final_train_loss": Average training loss of the final epoch.
            - "final_val_loss": Simulated validation loss of the final epoch.

    Example:
        >>> info = run_pytorch_experiment_mlflow(
        ...     model=model,
        ...     dataloader=dataloader,
        ...     criterion=criterion,
        ...     optimizer=torch.optim.Adam,
        ...     hidden_size=32,
        ...     experiment_name="pytorch_experiment",
        ...     epochs=20,
        ...     learning_rate=0.001,
        ... )
        >>> print("MLflow Run Info:", info)
    """
    # Ensure reproducibility of results.
    torch.manual_seed(42)
    np.random.seed(42)

    # Obtain MLflow experiment ID, creating the experiment if needed.
    experiment_id = create_or_get_experiment(experiment_name)
    
    # Instantiate the optimizer with model parameters and specified learning rate.
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
    run_info = {}
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log hyperparameters to MLflow.
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            # The following assume the existence of these global variables:
            "batch_size": batch_size,
            "input_size": input_size,
            "num_classes": num_classes
        })
        
        # Training loop.
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            # Iterate through training batches.
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(dataloader)
            
            # Simulate a validation loss for demonstration purposes.
            val_loss = avg_loss + random.uniform(0.0, 0.1)
            
            # Log training and validation losses for the current epoch.
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
            time.sleep(0.1)  # Simulate computation time per epoch.

        # Collect and return experiment run information.
        run_info = {
            "experiment_id": experiment_id,
            "run_id": run.info.run_id,
            "final_train_loss": avg_loss,
            "final_val_loss": val_loss
        }
    return run_info


if __name__ == "__main__":
    # Example usage:
    # Generate dummy data for binary classification.
    num_samples = 1000
    input_size = 10
    num_classes = 2
    batch_size = 32
    hidden_size = 32

    # Create dummy numpy arrays for features and labels.
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    
    # Import required classes for DataLoader and model.
    from torch.utils.data import DataLoader, TensorDataset
    from models.dl_models.simple_classifier import SimpleClassifier

    # Create a TensorDataset and DataLoader.
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, criterion, and specify optimizer class.
    model = SimpleClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam

    # Execute the experiment run with MLflow tracking.
    info = run_pytorch_experiment_mlflow(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        hidden_size=hidden_size,
        experiment_name="pytorch_experiment",
        epochs=20,
        learning_rate=0.001,
    )
    print("MLflow Run Info:", info)
