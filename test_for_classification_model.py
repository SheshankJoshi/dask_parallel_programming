from experiments.ml_experiments.run_experiments import classification_experiment
from dask.distributed import Client
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
# Load sample data (using the Iris dataset) and split it
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

if __name__ == "__main__":
    results = classification_experiment(scheduler_address="localhost:8786", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("Experiment Results:")
    for res in results:
        print(f"{res['model']} with params {res['params']} ->  achieved accuracy: {res['metrics']["accuracy"]:.4f}")