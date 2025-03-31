from dask.distributed import Client

from experiments.ml_experiments.setup_optimizations import grid_search_cv_and_refit
from sklearn.datasets import load_iris

def optimize_model(scheduler_address, model, X,y):
    client = Client(scheduler_address)
    data = load_iris()
    X, y = data.data, data.target

    param_grid = {
        "max_iter": [100, 200, 300],
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"]
    }
    base_model = model

    best_model, best_params, best_score = grid_search_cv_and_refit(
        client, base_model, param_grid, X, y, cv=5, scoring="accuracy"
    )

    return best_model, best_params, best_score