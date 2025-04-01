from experiments.ml_experiments.run_experiments import classification_experiment
from dask.distributed import Client
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from utils.client_utils.client_direct_folder_sync import UploadAndSetCWDPlugin
# Load sample data (using the Iris dataset) and split it
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


if __name__ == "__main__":
    # Connect to the Dask cluster (adjust the address as needed)
    # scheduler_address = "localhost:8786"
    # client = Client(scheduler_address)
    # print("Connected to Dask scheduler:", client)
    # # print(client.get_versions(check=True))
    # plugin = UploadAndSetCWDPlugin()
    # client.register_plugin(plugin)
    from dask_cuda import CUDAWorker
    from dask.distributed import LocalCluster
    cluster = LocalCluster()
    client = Client(cluster)
    print("Connected to Dask scheduler:", client)
    results = classification_experiment(client, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("Experiment Results:")
    for res in results:
        print(f"{res['model']} with params {res['params']} ->  achieved accuracy: {res['metrics']["accuracy"]:.4f}")