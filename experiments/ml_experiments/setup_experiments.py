import itertools
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


def run_experiment(model_class, params, X_train, X_test, y_train, y_test):
    """
    Train a single experiment using a model instantiated with given parameters,
    and evaluate its performance.
    For classifiers, multiple metrics are returned, and for regressors, regression
    metrics are computed.
    
    Args:
        model_class: scikit-learn model class.
        params: Dictionary of hyperparameters.
        X_train, X_test, y_train, y_test: Data for training and evaluation.
    
    Returns:
        A dict with:
           - model: Name of the model.
           - params: The used hyperparameters.
           - metrics: A dictionary with evaluation metrics.
    """
    # Instantiate and train the model
    model = model_class(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Determine evaluation metrics based on the model type
    if hasattr(model, "predict_proba"):
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro', zero_division=0)
        recall = recall_score(y_test, predictions, average='macro', zero_division=0)
        f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    else:
        # Regression metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "r2": r2,
            "mae": mae,
            "mse": mse
        }
    
    return {
        "model": model_class.__name__,
        "params": params,
        "metrics": metrics
    }

def generate_param_combinations(param_grid):
    """
    Given a param grid, generate all combinations.
    
    Args:
        param_grid: dict of parameter names and candidate value lists.
    
    Returns:
        A list of dicts representing all combinations.
    """
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, comb)) for comb in itertools.product(*values)]

def prepare_experiments(model_class, param_grid):
    """
    Prepare a list of experiments based on a parameter grid.
    
    Args:
        model_class: The scikit-learn model class.
        param_grid: Dictionary of parameters and candidate values.
    
    Returns:
        List of tuples (model_class, params) for each parameter combination.
    """
    experiments = []
    for params in generate_param_combinations(param_grid):
        experiments.append((model_class, params))
    return experiments


# ------------------------- Classification Utilities -------------------------
def get_all_classifiers():
    """
    Return a dict mapping classifier names to tuples of (model_class, param_grid)
    for many scikit-learn classifiers.
    """


    classifiers = {
        "LogisticRegression": (
            LogisticRegression,
            {"max_iter": [100, 200], "C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
        ),
        "SVC": (
            SVC,
            {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf", "poly"], "gamma": ["scale", "auto"]}
        ),
        "NuSVC": (
            NuSVC,
            {"nu": [0.1, 0.5, 0.9], "kernel": ["rbf", "linear"]}
        ),
        "LinearSVC": (
            LinearSVC,
            {"C": [0.1, 1.0, 10.0], "penalty": ["l2"], "loss": ["hinge", "squared_hinge"], "max_iter": [1000, 2000]}
        ),
        "RandomForestClassifier": (
            RandomForestClassifier,
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "criterion": ["gini", "entropy"]}
        ),
        "DecisionTreeClassifier": (
            DecisionTreeClassifier,
            {"max_depth": [None, 5, 10], "criterion": ["gini", "entropy"]}
        ),
        "KNeighborsClassifier": (
            KNeighborsClassifier,
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "algorithm": ["auto", "ball_tree", "kd_tree"]}
        ),
        "GaussianNB": (
            GaussianNB,
            {"var_smoothing": [1e-09, 1e-08, 1e-07]}
        ),
        "MultinomialNB": (
            MultinomialNB,
            {"alpha": [0.5, 1.0, 1.5], "fit_prior": [True, False]}
        ),
        "BernoulliNB": (
            BernoulliNB,
            {"alpha": [0.0, 0.5, 1.0], "binarize": [0.0, 0.5, 1.0], "fit_prior": [True, False]}
        ),
        "ComplementNB": (
            ComplementNB,
            {"alpha": [0.5, 1.0, 1.5], "norm": [True, False]}
        ),
        "AdaBoostClassifier": (
            AdaBoostClassifier,
            {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]}
        ),
        "GradientBoostingClassifier": (
            GradientBoostingClassifier,
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 1.0], "max_depth": [3, 5, 7]}
        ),
        "ExtraTreesClassifier": (
            ExtraTreesClassifier,
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "criterion": ["gini", "entropy"]}
        ),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier,
            {"max_iter": [100, 200], "learning_rate": [0.01, 0.1, 1.0], "max_depth": [None, 5, 10]}
        ),
        "LinearDiscriminantAnalysis": (
            LinearDiscriminantAnalysis,
            {"solver": ["svd", "lsqr"], "shrinkage": [None, "auto", 0.5]}
        ),
        "QuadraticDiscriminantAnalysis": (
            QuadraticDiscriminantAnalysis,
            {"reg_param": [0.0, 0.1, 0.2]}
        ),
        "Perceptron": (
            Perceptron,
            {"penalty": [None, "l2", "elasticnet"], "alpha": [0.0001, 0.001, 0.01], "max_iter": [1000, 2000]}
        ),
        "RidgeClassifier": (
            RidgeClassifier,
            {"alpha": [0.5, 1.0, 5.0], "solver": ["auto", "svd", "cholesky"]}
        ),
        "SGDClassifier": (
            SGDClassifier,
            {"loss": ["hinge", "log", "modified_huber", "squared_hinge"],
             "alpha": [0.0001, 0.001, 0.01],
             "max_iter": [1000, 2000]}
        ),
        "PassiveAggressiveClassifier": (
            PassiveAggressiveClassifier,
            {"C": [0.01, 0.1, 1.0, 10.0],
             "loss": ["hinge", "squared_hinge"],
             "max_iter": [1000, 2000]}
        ),
        "NearestCentroid": (
            NearestCentroid,
            {}  # No hyperparameters to tune by default.
        ),
        "CalibratedClassifierCV": (
            CalibratedClassifierCV,
            {"method": ["sigmoid", "isotonic"], "cv": [3, 5]}
        )
    }
    return classifiers

# ------------------------- Regression Utilities -------------------------
def get_all_regressors():
    """
    Return a dict mapping regressor names to tuples of (model_class, param_grid)
    for many scikit-learn regression models, including kernel-based methods.
    """

    regressors = {
        "LinearRegression": (
            LinearRegression,
            {}  # No hyperparameters to tune by default.
        ),
        "Ridge": (
            Ridge,
            {"alpha": [0.1, 1.0, 10.0], "solver": ["auto", "svd", "cholesky"], "max_iter": [None, 1000]}
        ),
        "Lasso": (
            Lasso,
            {"alpha": [0.01, 0.1, 1.0, 10.0], "max_iter": [1000, 5000]}
        ),
        "ElasticNet": (
            ElasticNet,
            {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9], "max_iter": [1000, 5000]}
        ),
        "BayesianRidge": (
            BayesianRidge,
            {"n_iter": [300, 500, 1000]}
        ),
        "SVR": (
            SVR,
            {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "poly", "rbf"],
             "gamma": ["scale", "auto"], "epsilon": [0.1, 0.2, 0.5]}
        ),
        "KernelRidge": (
            KernelRidge,
            {"alpha": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf", "polynomial"],
             "degree": [2, 3, 4]}  # degree is used for the "polynomial" kernel
        ),
        "DecisionTreeRegressor": (
            DecisionTreeRegressor,
            {"max_depth": [None, 5, 10, 20], "criterion": ["squared_error", "friedman_mse"]}
        ),
        "RandomForestRegressor": (
            RandomForestRegressor,
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "criterion": ["squared_error", "absolute_error"]}
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor,
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 1.0], "max_depth": [3, 5, 7]}
        ),
        "ExtraTreesRegressor": (
            ExtraTreesRegressor,
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "criterion": ["squared_error", "absolute_error"]}
        ),
        "AdaBoostRegressor": (
            AdaBoostRegressor,
            {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]}
        ),
        "KNeighborsRegressor": (
            KNeighborsRegressor,
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
        ),
        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor,
            {"max_iter": [100, 200], "learning_rate": [0.01, 0.1, 1.0], "max_depth": [None, 5, 10]}
        )
    }
    return regressors