#%%
import numpy as np
from dask.distributed import Client
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#%%

# Connect to the Dask cluster. This will connect to a running scheduler or start a local client.
client = Client("localhost:8786")
print("Connected to Dask scheduler:", client)

#%% Setting Training Loops

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Train the given model and evaluate its accuracy.
    
    Args:
        model: scikit-learn model instance
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    
    Returns:
        A tuple (model_name, accuracy) where accuracy is the score on the test set.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return type(model).__name__, accuracy

#%% Loading the Data

# Load sample data (using Iris dataset)
data = load_iris()
X, y = data.data, data.target
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%

# Initialize different models from scikit-learn
models = [
    LogisticRegression(max_iter=200),
    SVC(),
    RandomForestClassifier(n_estimators=100)
]

#%%

# Submit training tasks to the Dask cluster for each model
futures = [client.submit(train_and_evaluate, model, X_train, X_test, y_train, y_test) for model in models]
results = client.gather(futures)

print("Model performance:")
for model_name, accuracy in results:
    print(f"{model_name}: {accuracy:.4f}")
# %%
