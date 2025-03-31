import subprocess
import os

#! NOTE : All these are not guaranteed to work at all. Keep an eye out for them. There are much better alterntaives here.


def dask_setup_venv(venv_path):
    # Activate the specific Python environment
    activate_command = f"source {venv_path}/bin/activate"
    subprocess.run(activate_command, shell=True, check=True)

    # Optional: Set any necessary environment variables
    os.environ["YOUR_ENV_VARIABLE"] = "value"

    # Print a message to indicate that the environment has been activated
    print("Environment activated and setup completed.")


# === Example Setup of Conda 

# chmod + x setup_env.sh # First make sure that it is executable

def dask_setup_using_shell_script(shell_script_path):
    # Execute the shell script
    subprocess.run([shell_script_path], check=True)
    print("Shell script executed and environment setup completed.")
    

def dask_setup_conda(conda_env_name, env_vars={}):

    environment_not_activated=True

    # Check if any conda environment is active
    if 'CONDA_DEFAULT_ENV' in os.environ:
        active_env = os.environ['CONDA_DEFAULT_ENV']
        if active_env == conda_env_name:
            print(f"The Conda environment '{conda_env_name}' is already active.")
            environment_not_activated=False
        else:
            print(f"Currently active environment is '{active_env}'. Switching to the default environment first.")
    else:
        print("No Conda environment is currently active. Switching to the default environment first.")
        # Run 'conda init' before activating the default (base) environment
        subprocess.run("conda init", shell=True, check=True)
        print("Conda initialized.")
        # Switch to default (base) environment
        subprocess.run("conda activate base", shell=True, check=True)
        print("Base environment activated")
    if environment_not_activated:
        # Check if the requested conda environment exists
        conda_envs = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        if conda_env_name not in conda_envs.stdout:
            print(f"Conda environment '{conda_env_name}' does not exist.")
            raise ValueError(f"Conda environment '{conda_env_name}' does not exist.")

        # Activate the requested conda environment
        subprocess.run(f"conda activate {conda_env_name}", shell=True, check=True)
        print(f"Conda environment '{conda_env_name}' activated.")
        environment_not_activated=False
    
    if not environment_not_activated:
        # Set the additional environment variables, if any
        for key, value in env_vars.items():
            os.environ[key] = value
        print("Environment variables set.")
    
if __name__ == "__main__":
    # Example usage
    try:
        # dask_setup_venv("/path/to/your/venv")
        # dask_setup_using_shell_script("/path/to/your/setup_env.sh")
        dask_setup_conda("rapids-25.02")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")



