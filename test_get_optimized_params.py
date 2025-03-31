from experiments.ml_experiments.run_optimization import optimize_model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
X, y = data.data, data.target

if __name__ == "__main__":
    # Note : Here we are assuming that the best model is LogisticRegression
    best_model, best_params, best_score = optimize_model("localhost:8786", LogisticRegression(), X, y)
    print("Best Model:", best_model)
    print("Best Params:", best_params)
    print("Best CV Score:", best_score)