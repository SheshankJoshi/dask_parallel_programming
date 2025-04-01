# filepath: /media/sheshank/Work_Code/Work_folders/code/visualstudio/dask_parallel_programming/models/run_experiments.py

from dask.distributed import Client
from .setup_experiments import run_experiment, prepare_experiments, get_all_classifiers, get_all_regressors
from utils.client_utils.client_direct_folder_sync import UploadAndSetCWDPlugin

def classification_experiment(client, X_train, y_train, X_test, y_test):
    

    classifiers = get_all_classifiers()
    
    futures = []    
    # Loop through each classifier and its hyperparameter grid
    for name, (model_class, param_grid) in classifiers.items():
        experiments = prepare_experiments(model_class, param_grid)
        for m_class, params in experiments:
            future = client.submit(run_experiment, m_class, params, X_train, X_test, y_train, y_test)
            futures.append(future)
    
    # Collect and return results (or process them as needed)
    results = client.gather(futures)
    return results

def regression_experiment(client, X_train, y_train, X_test, y_test):
    # TODO : Client should be directly passed, not a scheduler address. That's the only way to avoid uncessary overhead in configurations. 

    
    regressors = get_all_regressors()
    futures = []
    
    for name, (model_class, param_grid) in regressors.items():
        experiments = prepare_experiments(model_class, param_grid)
        for m_class, params in experiments:
            future = client.submit(run_experiment, m_class, params, X_train, X_test, y_train, y_test)
            futures.append(future)
    
    results = client.gather(futures)
    return results

if __name__ == "__main__":
    scheduler_address = "localhost:8786"  # adjust if necessary
    results = classification_experiment(scheduler_address)
    for res in results:
        print(f"{res['model']} with params {res['params']} -> accuracy: {res['accuracy']:.4f}")
    


