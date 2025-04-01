"""
================================================================================
Simple Data Manager Module
--------------------------------------------------------------------------------
This module defines a simple data management framework for creating datasets and 
DataLoader objects for rapid experimentation and testing of models. It provides
classes that inherit from torch's Dataset and DataLoader functionalities and exposes
class methods to quickly generate or load data from various formats (e.g. NumPy arrays,
pandas DataFrames, CSV files) as well as publicly available datasets.

Main Components:
  • CustomDataset:
      - Inherits from torch.utils.data.Dataset.
      - Wraps input features (X) and targets (y) and supports different data types.
      
  • SimpleDataManager:
      - Contains default parameters (e.g. DEFAULT_BATCH_SIZE).
      - Provides class methods to:
          - Create CustomDataset instances from NumPy arrays, pandas DataFrames,
            or by generating random synthetic data.
          - Load data from CSV files.
          - Load publicly available datasets using scikit-learn, torchvision, 
            tf.keras, torchtext, or custom download methods.
          - Return ready-made DataLoader objects.
          
  • DatasetRegistry:
      - Maintains a registry of built-in datasets from various libraries (PyTorch,
        TensorFlow, scikit-learn, torchtext) along with metadata and task tags.
      - Tags datasets according to the task (e.g. classification, regression, vision, nlp, time_series).
      - Provides methods to list datasets by task, retrieve detailed metadata for a given dataset,
        and dynamically populate metadata by attempting to load the dataset.
      
Usage Example:
    >>> from managers.simple_data_manager import SimpleDataManager, DatasetRegistry
    >>> # Create a dataset from random NumPy arrays.
    >>> dataset = SimpleDataManager.from_numpy(num_samples=1000, input_size=10, num_classes=2)
    >>> dataloader = SimpleDataManager.get_dataloader(dataset)
    >>> for batch in dataloader:
    ...     X_batch, y_batch = batch
    ...     print(X_batch.shape, y_batch.shape)
    >>> # List all classification datasets.
    >>> classification_datasets = DatasetRegistry.list_datasets_for_task("classification")
    >>> print(classification_datasets)
    >>> # Dynamically update metadata for all registered datasets.
    >>> DatasetRegistry.populate_metadata()

Dependencies:
    - torch: For Dataset and DataLoader abstractions.
    - numpy: To generate and process array-based data.
    - pandas: For DataFrame based data inputs.
    - scikit-learn: To load public datasets (if available).
================================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

try:
    # scikit-learn dataset loaders
    from sklearn.datasets import load_iris, load_wine, load_digits, load_diabetes
except ImportError:
    load_iris = load_wine = load_digits = load_diabetes = None

# Try importing torchvision datasets.
try:
    from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
    import torchvision.transforms as transforms
except ImportError:
    MNIST = FashionMNIST = CIFAR10 = CIFAR100 = None

# Try importing TensorFlow Keras datasets.
try:
    from tensorflow.keras.datasets import mnist as tf_mnist, cifar10 as tf_cifar10, imdb as tf_imdb
except ImportError:
    tf_mnist = tf_cifar10 = tf_imdb = None

# Try importing torchtext datasets.
try:
    from torchtext.datasets import AG_NEWS, IMDB as TT_IMDB
except ImportError:
    AG_NEWS = TT_IMDB = None


class CustomDataset(Dataset):
    """
    CustomDataset wraps input data arrays or tensors and target labels.
    
    Parameters:
        X (np.ndarray or torch.Tensor): Input features.
        y (np.ndarray or torch.Tensor): Target values or labels.
    
    Both X and y will be converted to torch.Tensor if they are provided as NumPy arrays.
    """
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X)
        elif isinstance(X, torch.Tensor):
            self.X = X
        else:
            raise TypeError("X must be either a NumPy array or a torch.Tensor")
        
        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y)
        elif isinstance(y, torch.Tensor):
            self.y = y
        else:
            raise TypeError("y must be either a NumPy array or a torch.Tensor")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class TextCustomDataset(Dataset):
    """
    TextCustomDataset wraps text samples and their corresponding labels.
    
    Parameters:
        X (list): List of text samples (strings).
        y (list): List of target labels.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class SimpleDataManager:
    """
    SimpleDataManager provides ready-made methods to quickly generate and load data 
    for testing and training models.
    
    Class Variables:
        DEFAULT_BATCH_SIZE (int): Default batch size for DataLoader objects (default is 32).
    
    Available Class Methods:
        from_numpy(num_samples, input_size, num_classes):
            - Generate random data as NumPy arrays for testing.
        
        from_pandas(X_df, y_series):
            - Create a dataset from pandas DataFrame/Series.
        
        from_csv(csv_file, target_column, **read_csv_kwargs):
            - Create a dataset by reading a CSV file.
        
        get_dataloader(dataset, batch_size=None, shuffle=True):
            - Return a DataLoader for the given dataset.
        
        load_registered_dataset(dataset_name):
            - Load a dataset from the DatasetRegistry and return a Dataset object.
    """
    DEFAULT_BATCH_SIZE = 32

    @classmethod
    def from_numpy(cls, num_samples: int, input_size: int, num_classes: int) -> CustomDataset:
        X = np.random.randn(num_samples, input_size).astype(np.float32)
        y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
        return CustomDataset(X, y)
    
    @classmethod
    def from_pandas(cls, X_df: pd.DataFrame, y_series: pd.Series) -> CustomDataset:
        X = X_df.to_numpy().astype(np.float32)
        y = y_series.to_numpy().astype(np.int64)
        return CustomDataset(X, y)
    
    @classmethod
    def from_csv(cls, csv_file: str, target_column: str, **read_csv_kwargs) -> CustomDataset:
        df = pd.read_csv(csv_file, **read_csv_kwargs)
        X_df = df.drop(columns=[target_column])
        y_series = df[target_column]
        return cls.from_pandas(X_df, y_series)
    
    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int = None, shuffle: bool = True) -> DataLoader:
        if batch_size is None:
            batch_size = cls.DEFAULT_BATCH_SIZE
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @classmethod
    def load_registered_dataset(cls, dataset_name: str) -> Dataset:
        metadata = DatasetRegistry.get_dataset_metadata(dataset_name)
        if metadata is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry.")
        source = metadata.get("source").lower()
        dataset_name_lower = dataset_name.lower()
        if source == "sklearn":
            if dataset_name_lower == "iris":
                data = load_iris()
                X = np.array(data.data, dtype=np.float32)
                y = np.array(data.target, dtype=np.int64)
                return CustomDataset(X, y)
            elif dataset_name_lower == "wine":
                data = load_wine()
                X = np.array(data.data, dtype=np.float32)
                y = np.array(data.target, dtype=np.int64)
                return CustomDataset(X, y)
            elif dataset_name_lower == "digits":
                data = load_digits()
                X = np.array(data.data, dtype=np.float32)
                y = np.array(data.target, dtype=np.int64)
                return CustomDataset(X, y)
            elif dataset_name_lower == "diabetes":
                data = load_diabetes()
                X = np.array(data.data, dtype=np.float32)
                y = np.array(data.target, dtype=np.int64)
                return CustomDataset(X, y)
        elif source == "torchvision":
            transform = transforms.ToTensor()
            if dataset_name_lower == "mnist":
                return MNIST(root="data", train=True, download=True, transform=transform)
            elif dataset_name_lower == "fashionmnist":
                return FashionMNIST(root="data", train=True, download=True, transform=transform)
            elif dataset_name_lower == "cifar10":
                return CIFAR10(root="data", train=True, download=True, transform=transform)
            elif dataset_name_lower == "cifar100":
                return CIFAR100(root="data", train=True, download=True, transform=transform)
        elif source == "tf.keras":
            if dataset_name_lower == "tf_mnist":
                (X_train, y_train), _ = tf_mnist.load_data()
                X_train = X_train.astype(np.float32) / 255.0
                y_train = y_train.astype(np.int64)
                # Add channel dimension if necessary
                if len(X_train.shape) == 3:
                    X_train = np.expand_dims(X_train, axis=1)
                return CustomDataset(X_train, y_train)
            elif dataset_name_lower == "tf_cifar10":
                (X_train, y_train), _ = tf_cifar10.load_data()
                X_train = X_train.astype(np.float32) / 255.0
                y_train = y_train.flatten().astype(np.int64)
                return CustomDataset(X_train, y_train)
            elif dataset_name_lower == "imdb":
                (X_train, y_train), _ = tf_imdb.load_data(num_words=10000)
                return TextCustomDataset(X_train, y_train)
        elif source == "torchtext":
            if dataset_name_lower == "ag_news":
                if AG_NEWS is None:
                    raise ImportError("torchtext not installed. Install via: pip install torchtext")
                train_iter = AG_NEWS(split='train')
                texts, labels = [], []
                for label, text in train_iter:
                    texts.append(text)
                    labels.append(label)
                return TextCustomDataset(texts, labels)
            elif dataset_name_lower == "imdb_text":
                if TT_IMDB is None:
                    raise ImportError("torchtext not installed. Install via: pip install torchtext")
                train_iter = TT_IMDB(split='train')
                texts, labels = [], []
                for label, text in train_iter:
                    texts.append(text)
                    labels.append(label)
                return TextCustomDataset(texts, labels)
        elif source == "download":
            raise NotImplementedError("Custom download logic is not implemented yet.")
        else:
            raise ValueError(f"Source '{source}' is not supported.")
        raise ValueError(f"Dataset '{dataset_name}' could not be loaded.")


class DatasetRegistry:
    """
    DatasetRegistry maintains a registry of built-in datasets from various libraries 
    (PyTorch, TensorFlow, scikit-learn, torchtext, etc.) with metadata and task tags.
    Datasets include vision, NLP, classification, regression, and time series data.

    Each dataset entry is a dictionary with keys:
        - name: Name of the dataset.
        - source: The library source (e.g., 'sklearn', 'torchvision', 'tf.keras', 'torchtext').
        - task: Task tag(s) (e.g., 'classification', 'regression', 'vision', 'nlp', 'time_series').
        - num_rows (optional): Total number of samples (if known).
        - num_features (optional): Number of features per sample.
        - target_column (optional): Name or index of the target column.
        - feature_columns (optional): List of feature column names (if applicable).
        - description: Short description of the dataset.
    
    Available Class Methods:
        list_datasets_for_task(task: str) -> list:
            - List all registered datasets matching the specified task.
        
        get_dataset_metadata(dataset_name: str) -> dict or None:
            - Retrieve metadata for a given dataset by name.

        populate_metadata() -> None:
            - Dynamically update missing metadata for each registered dataset if possible.
              If a required loader is missing, it informs the user how to install the package.
    """
    _datasets = [
        # Sklearn Datasets
        {
            "name": "Iris",
            "source": "sklearn",
            "task": "classification",
            "description": "Iris flower classification dataset (3 classes)."
        },
        {
            "name": "Wine",
            "source": "sklearn",
            "task": "classification",
            "description": "Wine recognition dataset used for classification."
        },
        {
            "name": "Digits",
            "source": "sklearn",
            "task": "classification, vision",
            "description": "Handwritten digits for image classification."
        },
        {
            "name": "Diabetes",
            "source": "sklearn",
            "task": "regression",
            "description": "Diabetes dataset for regression tasks."
        },
        # Torchvision Datasets
        {
            "name": "MNIST",
            "source": "torchvision",
            "task": "classification, vision",
            "description": "Handwritten digit dataset for image classification."
        },
        {
            "name": "FashionMNIST",
            "source": "torchvision",
            "task": "classification, vision",
            "description": "Fashion MNIST for clothing item classification."
        },
        {
            "name": "CIFAR10",
            "source": "torchvision",
            "task": "classification, vision",
            "description": "CIFAR10 dataset (10 classes, 32x32 images)."
        },
        {
            "name": "CIFAR100",
            "source": "torchvision",
            "task": "classification, vision",
            "description": "CIFAR100 dataset (100 classes, 32x32 images)."
        },
        # TensorFlow Keras Datasets
        {
            "name": "TF_MNIST",
            "source": "tf.keras",
            "task": "classification, vision",
            "description": "TensorFlow Keras version of the MNIST dataset."
        },
        {
            "name": "TF_CIFAR10",
            "source": "tf.keras",
            "task": "classification, vision",
            "description": "TensorFlow Keras version of the CIFAR10 dataset."
        },
        {
            "name": "IMDB",
            "source": "tf.keras",
            "task": "classification, nlp",
            "description": "IMDB movie reviews for sentiment analysis."
        },
        {
            "name": "Reuters",
            "source": "tf.keras",
            "task": "classification, nlp",
            "description": "Reuters newswire topics dataset for classification."
        },
        # Torchtext Datasets
        {
            "name": "AG_NEWS",
            "source": "torchtext",
            "task": "classification, nlp",
            "description": "AG News dataset for text classification."
        },
        {
            "name": "IMDB_Text",
            "source": "torchtext",
            "task": "classification, nlp",
            "description": "IMDB reviews for sentiment analysis (torchtext version)."
        },
        # Synthetic / External Time Series
        {
            "name": "Synthetic_TimeSeries",
            "source": "synthetic",
            "task": "time_series, regression",
            "description": "Randomly generated time series data for regression."
        },
        {
            "name": "ElectricityLoadDiagrams",
            "source": "download",
            "task": "time_series, regression",
            "description": "Electricity load diagrams dataset from UCI repository."
        }
    ]

    @classmethod
    def list_datasets_for_task(cls, task: str) -> list:
        task_lower = task.lower()
        return [ds for ds in cls._datasets if task_lower in ds["task"].lower()]
    
    @classmethod
    def get_dataset_metadata(cls, dataset_name: str) -> dict:
        for ds in cls._datasets:
            if ds["name"].lower() == dataset_name.lower():
                return ds
        return None

    @classmethod
    def populate_metadata(cls) -> None:
        """
        Dynamically populate missing metadata for each registered dataset.
        For each dataset based on its source, attempt to load it (or fetch info)
        and update keys such as num_rows, num_features, target_column, and feature_columns.
        If the required package is missing, print instructions to install it.
        """
        for ds in cls._datasets:
            source = ds.get("source", "").lower()
            name = ds.get("name", "").lower()

            if source == "sklearn":
                try:
                    if name == "iris":
                        data = load_iris()
                        ds["num_rows"] = len(data.data)
                        ds["num_features"] = data.data.shape[1]
                        ds["target_column"] = "target"
                        ds["feature_columns"] = data.feature_names
                    elif name == "wine":
                        data = load_wine()
                        ds["num_rows"] = len(data.data)
                        ds["num_features"] = data.data.shape[1]
                        ds["target_column"] = "target"
                    elif name == "digits":
                        data = load_digits()
                        ds["num_rows"] = len(data.data)
                        ds["num_features"] = data.data.shape[1]
                        ds["target_column"] = "target"
                    elif name == "diabetes":
                        data = load_diabetes()
                        ds["num_rows"] = len(data.data)
                        ds["num_features"] = data.data.shape[1]
                        ds["target_column"] = "target"
                except ImportError:
                    print("scikit-learn not installed. Install via: pip install scikit-learn")
            elif source == "torchvision":
                try:
                    transform = transforms.ToTensor()
                    if name == "mnist":
                        dataset_obj = MNIST(root="data", train=True, download=True, transform=transform)
                        ds["num_rows"] = len(dataset_obj)
                        ds["num_features"] = "1x28x28"
                        ds["target_column"] = "label"
                    elif name == "fashionmnist":
                        dataset_obj = FashionMNIST(root="data", train=True, download=True, transform=transform)
                        ds["num_rows"] = len(dataset_obj)
                        ds["num_features"] = "1x28x28"
                        ds["target_column"] = "label"
                    elif name == "cifar10":
                        dataset_obj = CIFAR10(root="data", train=True, download=True, transform=transform)
                        ds["num_rows"] = len(dataset_obj)
                        ds["num_features"] = "3x32x32"
                        ds["target_column"] = "label"
                    elif name == "cifar100":
                        dataset_obj = CIFAR100(root="data", train=True, download=True, transform=transform)
                        ds["num_rows"] = len(dataset_obj)
                        ds["num_features"] = "3x32x32"
                        ds["target_column"] = "label"
                except Exception as e:
                    print(f"Torchvision dataset loading error for {ds['name']}: {e}")
                    print("Install torchvision via: pip install torchvision")
            elif source == "tf.keras":
                try:
                    if name == "tf_mnist":
                        (X_train, _), _ = tf_mnist.load_data()
                        ds["num_rows"] = X_train.shape[0]
                        ds["num_features"] = X_train.shape[1:]
                        ds["target_column"] = "label"
                    elif name == "tf_cifar10":
                        (X_train, _), _ = tf_cifar10.load_data()
                        ds["num_rows"] = X_train.shape[0]
                        ds["num_features"] = X_train.shape[1:]
                        ds["target_column"] = "label"
                    elif name == "imdb":
                        ds["num_rows"] = "variable"
                        ds["target_column"] = "sentiment"
                    elif name == "reuters":
                        ds["num_rows"] = "variable"
                        ds["target_column"] = "topic"
                except Exception as e:
                    print(f"TensorFlow Keras dataset error for {ds['name']}: {e}")
                    print("Install tensorflow via: pip install tensorflow")
            elif source == "torchtext":
                # For torchtext datasets, we set metadata as 'variable' or leave it.
                if name in ["ag_news", "imdb_text"]:
                    ds["num_rows"] = "variable"
                    ds["target_column"] = "label"
            elif source == "synthetic":
                # We assume synthetic datasets already have metadata.
                pass
            elif source == "download":
                # For external download datasets, we simply notify the user.
                print(f"Dataset '{ds['name']}' requires manual download. Check documentation for instructions.")
            else:
                print(f"Source '{source}' not recognized for dataset '{ds['name']}'.")

# -----------------------------------------------------------------------------
# Example Usage:
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Existing SimpleDataManager example:
    dataset = SimpleDataManager.from_numpy(num_samples=1000, input_size=10, num_classes=2)
    dataloader = SimpleDataManager.get_dataloader(dataset, batch_size=64)
    for X_batch, y_batch in dataloader:
        print("Random Dataset - X_batch shape:", X_batch.shape)
        print("Random Dataset - y_batch shape:", y_batch.shape)
        break

    # List all classification datasets.
    classification_datasets = DatasetRegistry.list_datasets_for_task("classification")
    print("\nClassification Datasets:")
    for ds in classification_datasets:
        print(ds["name"], "-", ds["description"])

    # Get metadata for the Iris dataset.
    iris_metadata = DatasetRegistry.get_dataset_metadata("Iris")
    print("\nIris Dataset Metadata:")
    print(iris_metadata)

    # Dynamically populate metadata for all registered datasets.
    print("\nPopulating metadata for registered datasets:")
    DatasetRegistry.populate_metadata()

    # Print updated metadata for Iris.
    iris_metadata = DatasetRegistry.get_dataset_metadata("Iris")
    print("\nUpdated Iris Dataset Metadata:")
    print(iris_metadata)
