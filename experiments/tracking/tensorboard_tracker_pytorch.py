"""
================================================================================
TensorBoard Tracker PyTorch Module
--------------------------------------------------------------------------------
This module provides a comprehensive framework for tracking PyTorch experiments 
using TensorBoard. It is designed to support a wide range of features for experiment
tracking, including logging of hyperparameters, scalar metrics, weight and gradient
histograms, and the computation graph.

Features:
  - Creates a TensorBoard SummaryWriter with a unique logging directory.
  - Logs hyperparameters using the add_hparams API.
  - Tracks training and (simulated) validation losses as scalar metrics.
  - Logs model weights and gradient histograms periodically.
  - Optionally logs the learning rate and other experiment details.
  - Provides a complete training loop using a PyTorch model, DataLoader, loss criterion,
    and optimizer.

Usage Example:
  >>> from torch.utils.data import DataLoader, TensorDataset
  >>> from models.dl_models.simpleNet_classifier import SimpleClassifier
  >>> import torch.nn as nn
  >>> import torch.optim as optim
  >>> 
  >>> # Prepare dummy data and dataloader.
  >>> dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
  >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  >>> 
  >>> # Instantiate model, criterion, and optimizer class.
  >>> model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=2)
  >>> criterion = nn.CrossEntropyLoss()
  >>> optimizer_class = optim.Adam
  >>> 
  >>> # Define hyperparameters.
  >>> hparams = {
  ...    "input_size": 10,
  ...    "hidden_size": 32,
  ...    "num_classes": 2,
  ...    "batch_size": 32,
  ...    "epochs": 20,
  ...    "learning_rate": 0.001
  ... }
  >>> 
  >>> # Run the experiment.
  >>> run_info = run_pytorch_experiment_tb(
  ...    model=model,
  ...    dataloader=dataloader,
  ...    criterion=criterion,
  ...    optimizer_class=optimizer_class,
  ...    experiment_name="tensorboard_experiment",
  ...    epochs=20,
  ...    learning_rate=0.001,
  ...    log_interval=1,
  ...    device="cpu",
  ...    hparams=hparams,
  ... )
  >>> print("TensorBoard Experiment Run Info:", run_info)

Dependencies:
  - torch, numpy: For model training and numerical computations.
  - torch.utils.tensorboard: For logging to TensorBoard.
  - os, time, random: For file management, timing, and simulation purposes.
================================================================================
"""

import os
import time
import random
import numpy as np
import torch
from typing import Type
from torch.utils.tensorboard import SummaryWriter

def create_tb_writer(experiment_name: str, log_dir: str = None) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter for experiment tracking.
    
    Parameters:
        experiment_name (str): Name of the experiment.
        log_dir (str, optional): Directory to store TensorBoard logs. If None, defaults to './tensorboard_logs'.
    
    Returns:
        SummaryWriter: A TensorBoard writer instance with a log directory based on the experiment name and timestamp.
    """
    if log_dir is None:
        log_dir = "./tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    writer = SummaryWriter(log_dir=experiment_path)
    print(f"TensorBoard logs will be written to: {experiment_path}")
    return writer

def run_pytorch_experiment_tb(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    experiment_name: str = "tensorboard_experiment",
    epochs: int = 20,
    learning_rate: float = 0.001,
    log_interval: int = 1,
    device: str = "cpu",
    hparams: dict = None,
) -> dict:
    """
    Run a PyTorch experiment and track metrics using TensorBoard.
    
    This function performs the following:
      - Sets up a training loop with the provided model, dataloader, loss criterion, and optimizer.
      - Logs hyperparameters using the add_hparams API.
      - Tracks scalar metrics (train and validation losses, learning rate) at every epoch.
      - Logs histograms of weights and gradients for the first batch of each epoch.
      - Optionally logs additional experiment details and the computation duration.
    
    Parameters:
      model (torch.nn.Module): The neural network model to train.
      dataloader (torch.utils.data.DataLoader): Provides training data.
      criterion (torch.nn.Module): Loss function.
      optimizer_class (torch.optim.Optimizer): Optimizer class to use (e.g., torch.optim.Adam).
      experiment_name (str): Name for the experiment, used for naming the log directory.
      epochs (int): Number of training epochs.
      learning_rate (float): Learning rate for the optimizer.
      log_interval (int): Frequency in epochs to log weight and gradient histograms.
      device (str): Device to run training on (e.g., "cpu" or "cuda").
      hparams (dict): Hyperparameters dictionary; if None, defaults are used.
    
    Returns:
      dict: A summary containing final training/validation loss and total training time.
    """
    # For reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(device)
    model.to(device)

    writer = create_tb_writer(experiment_name)
    
    if hparams is None:
        hparams = {"epochs": epochs, "learning_rate": learning_rate}
    # Log hyperparameters using add_hparams.
    writer.add_hparams(hparams=hparams, metric_dict={})
    
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Log histograms for weights and gradients of the first batch each epoch.
            if i == 0 and epoch % log_interval == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Weights/{name}", param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
        
        avg_train_loss = running_loss / len(dataloader)
        # Simulate a validation loss for demonstration.
        avg_val_loss = avg_train_loss + random.uniform(0.0, 0.1)
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Learning_Rate", learning_rate, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    total_time = time.time() - start_time
    writer.add_text("Experiment_Details", f"Training completed in {total_time:.2f} seconds", epochs)
    writer.close()
    
    run_info = {
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "training_time": total_time,
    }
    return run_info

if __name__ == "__main__":
    # Example usage: Dummy data for a classification problem.
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from models.dl_models.simple_classifier import SimpleClassifier
    import torch.nn as nn
    import torch.optim as optim

    num_samples = 1000
    input_size = 10
    num_classes = 2
    batch_size = 32
    hidden_size = 32

    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_class = optim.Adam

    hparams = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "epochs": 20,
        "learning_rate": 0.001
    }

    run_info = run_pytorch_experiment_tb(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer_class=optimizer_class,
        experiment_name="tensorboard_experiment",
        epochs=20,
        learning_rate=0.001,
        log_interval=1,
        device="cpu",
        hparams=hparams,
    )
    print("TensorBoard Experiment Run Info:", run_info)