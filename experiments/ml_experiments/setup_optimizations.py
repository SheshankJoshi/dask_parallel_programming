import itertools
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone

def grid_search_cv(client, model, param_grid, X, y, cv=5, scoring="accuracy"):
    """
    Perform a distributed grid search cross-validation using Dask.

    Parameters:
        client         : Dask client to submit tasks.
        model          : scikit-learn estimator.
        param_grid     : dict; keys are parameter names, and values are lists of possible values.
        X              : Training data features.
        y              : Target array.
        cv             : Number of cross-validation folds (int or CV splitter).
        scoring        : Scoring metric (string or callable).

    Returns:
        best_params (dict): Parameter setting that achieved the best average score.
        best_score  (float): Best average cross-validation score.
    """
    # Create all hyperparameter combinations
    param_names = list(param_grid.keys())
    all_combinations = list(itertools.product(*(param_grid[name] for name in param_names)))
    
    # List to hold futures along with the corresponding param_dict.
    future_results = []
    
    for comb in all_combinations:
        params = dict(zip(param_names, comb))
        # Clone the model and set the hyperparameters.
        model_instance = clone(model)
        model_instance.set_params(**params)
        # Submit cross-validation computation to a Dask worker.
        # Using KFold so that our grid search is reproducible.
        future = client.submit(
            cross_val_score, 
            model_instance, 
            X, 
            y, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        )
        future_results.append((params, future))
    
    # Gather and determine the best parameters.
    best_score = -np.inf
    best_params = None
    
    for params, future in future_results:
        scores = future.result()
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score

def grid_search_cv_and_refit(client, model, param_grid, X, y, cv=5, scoring="accuracy"):
    """
    Perform grid search CV to find best hyperparameters and refit the model on the entire data.

    Parameters:
        client         : Dask client to submit tasks.
        model          : scikit-learn estimator.
        param_grid     : dict; keys are parameter names, and values are lists of possible values.
        X              : Training data features.
        y              : Target array.
        cv             : Number of cross-validation folds.
        scoring        : Scoring metric.

    Returns:
        best_model: The fitted model using the best hyperparameters.
        best_params (dict): Best hyperparameters.
        best_score  (float): Best CV score.
    """
    best_params, best_score = grid_search_cv(client, model, param_grid, X, y, cv=cv, scoring=scoring)
    # Clone a new model instance, set the best parameters and fit it on full training set
    best_model = clone(model)
    best_model.set_params(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score