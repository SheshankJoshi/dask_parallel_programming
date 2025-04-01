from dask.distributed import Client
from experiments.ml_experiments.run_experiments import regression_experiment
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from utils.client_utils.client_direct_folder_sync import UploadAndSetCWDPlugin
# Load a regression dataset.
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

if __name__ == "__main__":
    # Connect to the Dask cluster (adjust the address as needed)
    # scheduler_address = "localhost:8786"
    # client = Client(scheduler_address)
    # print("Connected to Dask scheduler:", client)
    # # print(client.get_versions(check=True))
    # plugin = UploadAndSetCWDPlugin()
    # client.register_plugin(plugin)
    from dask.distributed import LocalCluster
    # cluster = LocalCluster()
    client = Client()
    print("Connected to Dask scheduler:", client)
    reg_results = regression_experiment(client, X_train, y_train, X_test, y_test)
    print("Regression Experiment Results:")
    for res in reg_results:
        print(f"{res['model']} with params {res['params']} -> R^2: {res['metrics']["r2"]:.4f}")