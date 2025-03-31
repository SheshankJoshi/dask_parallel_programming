import os
import subprocess
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from typing import Optional, Dict, Union, List
from .asyncssh_config_models import AsyncSSHConnectOptions
import re
import dask
dask.config

# Allowed worker classes defined externally.
ALLOWED_WORKER_CLASSES = ["distributed.Nanny", "dask_cuda.CUDAWorker"]

class SchedulerCLIOptions(BaseModel):
    host: Optional[str] = Field(
        None,
        description="URI, IP, or hostname of this server."
    )
    port: Optional[int] = Field(
        None,
        description="Serving port."
    )
    interface: Optional[str] = Field(
        None,
        description="Preferred network interface like 'eth0' or 'ib0'."
    )
    protocol: Optional[str] = Field(
        None,
        description="Protocol like tcp, tls, or ucx."
    )
    tls_ca_file: Optional[str] = Field(
        None,
        description="CA cert(s) file for TLS (in PEM format)."
    )
    tls_cert: Optional[str] = Field(
        None,
        description="Certificate file for TLS (in PEM format)."
    )
    tls_key: Optional[str] = Field(
        None,
        description="Private key file for TLS (in PEM format)."
    )
    dashboard_address: Optional[str] = Field(
        ":8787",
        description="Address on which to listen for diagnostics dashboard."
    )
    dashboard: Optional[bool] = Field(
        True,
        description="Launch the Dashboard [default: True]."
    )
    jupyter: Optional[bool] = Field(
        False,
        description="Start a Jupyter Server in the same process."
    )
    show: Optional[bool] = Field(
        True,
        description="Show web UI [default: True]."
    )
    dashboard_prefix: Optional[str] = Field(
        None,
        description="Prefix for the dashboard app."
    )
    use_xheaders: Optional[bool] = Field(
        False,
        description="Use xheaders in dashboard app for SSL termination in header."
    )
    pid_file: Optional[str] = Field(
        None,
        description="File to write the process PID."
    )
    scheduler_file: Optional[str] = Field(
        None,
        description="File to write connection information."
    )
    preload: Optional[Union[str, list]] = Field(
        None,
        description="Module(s) that should be loaded by the scheduler process."
    )
    idle_timeout: Optional[str] = Field(
        None,
        description="Time of inactivity after which to kill the scheduler."
    )

    @field_validator("port")
    def validate_port(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < 1 or value > 65535):
            raise ValueError("Port must be between 1 and 65535.")
        return value

    @field_validator("dashboard_address")
    def validate_dashboard_address(cls, value: str) -> str:
        if not value:
            raise ValueError("dashboard_address must not be empty.")
        return value

    @field_validator("idle_timeout")
    def validate_idle_timeout(cls, value: Optional[str]) -> Optional[str]:
        if value and not value.endswith(("s", "m", "h")):
            raise ValueError(
                "idle_timeout must end with a valid time unit (e.g., 's', 'm', 'h')."
            )
        try:
            if value:
                int(value[:-1])  # Check if the numeric part is valid
        except ValueError:
            raise ValueError(
                "idle_timeout must start with a numeric value (e.g., '60s')."
            )
        return value


class WorkerCLIOptions(BaseModel):
    tls_ca_file: Optional[str] = Field(
        None,
        description="CA cert(s) file for TLS (in PEM format)."
    )
    tls_cert: Optional[str] = Field(
        None,
        description="Certificate file for TLS (in PEM format)."
    )
    tls_key: Optional[str] = Field(
        None,
        description="Private key file for TLS (in PEM format)."
    )
    worker_port: Optional[Union[int, str]] = Field(
        None,
        description=(
            "Serving computation port. Can be a single port or a range like '3000:3026'."
        )
    )
    nanny_port: Optional[Union[int, str]] = Field(
        None,
        description=(
            "Serving nanny port. Can be a single port or a range like '3000:3026'."
        )
    )
    scheduler_file: Optional[str] = Field(
        None,
        description="Filename to JSON encoded scheduler information."
    )
    pid_file: Optional[str] = Field(
        None,
        description="File to write the process PID."
    )
    local_directory: Optional[str] = Field(
        None,
        description="Directory to place worker files."
    )
    dashboard_address: Optional[str] = Field(
        ":8787",
        description="Address on which to listen for diagnostics dashboard."
    )
    dashboard: Optional[bool] = Field(
        True,
        description="Launch the Dashboard [default: True]."
    )
    listen_address: Optional[str] = Field(
        None,
        description="The address to which the worker binds."
    )
    contact_address: Optional[str] = Field(
        None,
        description="The address the worker advertises to the scheduler."
    )
    host: Optional[str] = Field(
        None,
        description="Serving host. Should be an IP address visible to the scheduler and other workers."
    )
    interface: Optional[str] = Field(
        None,
        description="Network interface like 'eth0' or 'ib0'."
    )
    protocol: Optional[str] = Field(
        None,
        description="Protocol like tcp, tls, or ucx."
    )
    nthreads: Optional[int] = Field(
        None,
        description="Number of threads per process."
    )
    nworkers: Optional[Union[int, str]] = Field(
        None,
        description=(
            "Number of worker processes to launch. Can be an integer or 'auto' for dynamic configuration."
        )
    )
    # The following is the alias config for the nworkers. Might have to be removed in futures depending on version upates
    n_workers: Optional[Union[int, str]] = Field(
        None,
        description=(
            "Number of worker processes to launch. Can be an integer or 'auto' for dynamic configuration."
        )
    ) # This is 
    name: Optional[str] = Field(
        None,
        description="A unique name for this worker."
    )
    memory_limit: Optional[Union[int, float, str]] = Field(
        "auto",
        description=(
            "Bytes of memory per process that the worker can use. Can be an integer (bytes), "
            "a float (fraction of total system memory), a string (e.g., '5GB'), or 'auto'."
        )
    )
    nanny: Optional[bool] = Field(
        True,
        description="Start workers in nanny process for management [default: True]."
    )
    resources: Optional[Dict[str, Union[int, float]]] = Field(
        None,
        description="Resources for task constraints like 'GPU=2 MEM=10e9'."
    )
    death_timeout: Optional[int] = Field(
        None,
        description="Seconds to wait for a scheduler before closing."
    )
    dashboard_prefix: Optional[str] = Field(
        None,
        description="Prefix for the dashboard."
    )
    lifetime: Optional[str] = Field(
        None,
        description="Shut down the worker after this duration."
    )
    lifetime_stagger: Optional[str] = Field(
        None,
        description="Random amount by which to stagger lifetime values."
    )
    lifetime_restart: Optional[bool] = Field(
        False,
        description="Whether to restart the worker after the lifetime lapses."
    )
    preload: Optional[Union[str, List[str]]] = Field(
        None,
        description="Module(s) to be loaded by each worker process."
    )
    preload_nanny: Optional[Union[str, List[str]]] = Field(
        None,
        description="Module(s) to be loaded by each nanny."
    )
    scheduler_sni: Optional[str] = Field(
        None,
        description="Scheduler SNI (if different from scheduler hostname)."
    )

    @field_validator("worker_port", "nanny_port", mode="before")
    def validate_port_range(cls, value: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        if isinstance(value, str) and ":" in value:
            start, end = value.split(":")
            if not start.isdigit() or not end.isdigit():
                raise ValueError("Port range must be in the format '<start>:<end>' with numeric values.")
            if int(start) > int(end):
                raise ValueError("Port range start must be less than or equal to end.")
        elif isinstance(value, int) and (value < 1 or value > 65535):
            raise ValueError("Port must be between 1 and 65535.")
        return value

    @field_validator("memory_limit", mode="before")
    def validate_memory_limit(cls, value: Optional[Union[int, float, str]]) -> Optional[Union[int, float, str]]:
        if isinstance(value, str) and value != "auto":
            if not value[-1].isdigit() and value[-1] not in "BKMGT":
                raise ValueError("Memory limit must be a valid size string (e.g., '5GB', '5000M') or 'auto'.")
        return value

    @field_validator("scheduler_file")
    def validate_scheduler_file(cls, value: Optional[str]) -> Optional[str]:
        if value and not value.endswith(".json"):
            raise ValueError("scheduler_file must have a .json file extension.")
        # Syntax validation: Ensure it's a valid file path
        if not re.match(r"^[\w\-/\.]+$", value):
            raise ValueError("scheduler_file contains invalid characters.")
        return value

    @field_validator("pid_file")
    def validate_pid_file(cls, value: Optional[str]) -> Optional[str]:
        if value and not value.endswith(".pid"):
            raise ValueError("pid_file must have a .pid file extension.")
        # Syntax validation: Ensure it's a valid file path
        if not re.match(r"^[\w\-/\.]+$", value):
            raise ValueError("pid_file contains invalid characters.")
        return value

    @field_validator("local_directory")
    def validate_local_directory(cls, value: Optional[str]) -> Optional[str]:
        # Syntax validation: Ensure it's a valid directory path
        if value and not re.match(r"^[\w\-/\.]+$", value):
            raise ValueError("local_directory contains invalid characters.")
        return value


class SchedulerConfig(BaseModel):
    address: str = Field(
        ...,
        description="The hostname where the scheduler should run."
    )
    connect_options: Optional[Union["AsyncSSHConnectOptions", Dict[str, str]]] = Field(
        None,
        description=(
            "Options for establishing an SSH connection. Can be either an "
            "AsyncSSHConnectOptions object or a dictionary of connection parameters."
        )
    )
    remote_python: str = Field(
        ...,
        description="Path to Python on the remote node to run this scheduler."
    )
    scheduler_timeout: Optional[str] = Field(
        "60s",
        description="Timeout for the scheduler (e.g., '60s', '5m')."
    )
    kwargs: Optional["SchedulerCLIOptions"] = Field(
        None,
        description="Additional arguments to be passed to the Dask scheduler CLI."
    )

    @field_validator("scheduler_timeout")
    def validate_scheduler_timeout(cls, value: str) -> str:
        if not value.endswith(("s", "m", "h")):
            raise ValueError(
                "scheduler_timeout must end with a valid time unit (e.g., 's', 'm', 'h')."
            )
        try:
            int(value[:-1])  # Check if the numeric part is valid
        except ValueError:
            raise ValueError(
                "scheduler_timeout must start with a numeric value (e.g., '60s')."
            )
        return value

    @field_validator("address")
    def validate_address(cls, value: str) -> str:
        if not value:
            raise ValueError("address must not be empty.")
        return value

    @field_validator("remote_python")
    def validate_remote_python(cls, value: str) -> str:
        if not value:
            raise ValueError("remote_python must not be empty.")
        return value

    @field_validator("connect_options", mode="before")
    def validate_connect_options(cls, value: Optional[Union["AsyncSSHConnectOptions", Dict[str, str]]]) -> Optional[Union["AsyncSSHConnectOptions", Dict[str, str]]]:
        if isinstance(value, dict):
            # Validate that the dictionary contains only string keys and values
            for key, val in value.items():
                if not isinstance(key, str) or not isinstance(val, str):
                    raise ValueError("All keys and values in connect_options dictionary must be strings.")
        return value


class WorkerConfig(BaseModel):
    scheduler: str = Field(
        ...,
        description="The address of the scheduler."
    )
    address: str = Field(
        ...,
        description="The hostname where the worker should run."
    )
    worker_class: Optional[str] = Field(
        None,
        description=(
            "The Python class to use to create the worker. Allowed values: "
            f"{ALLOWED_WORKER_CLASSES}"
        )
    )
    connect_options: Optional[Union["AsyncSSHConnectOptions", Dict[str, str]]] = Field(
        None,
        description=(
            "Options for establishing an SSH connection. Can be either an "
            "AsyncSSHConnectOptions object or a dictionary of connection parameters."
        )
    )
    remote_python: str = Field(
        ...,
        description="Path to Python on the remote node to run this worker."
    )
    kwargs: Optional[Union[Dict[str, Union[str, int, bool]], "WorkerCLIOptions"]] = Field(
        None,
        description="Additional arguments to be passed to the Dask worker CLI."
    )

    @field_validator("scheduler")
    def validate_scheduler(cls, value: str) -> str:
        if not value:
            raise ValueError("scheduler must not be empty.")
        return value

    @field_validator("address")
    def validate_address(cls, value: str) -> str:
        if not value:
            raise ValueError("address must not be empty.")
        return value

    @field_validator("remote_python")
    def validate_remote_python(cls, value: str) -> str:
        if not value:
            raise ValueError("remote_python must not be empty.")
        return value

    @field_validator("connect_options", mode="before")
    def validate_connect_options(cls, value: Optional[Union["AsyncSSHConnectOptions", Dict[str, str]]]) -> Optional[Union["AsyncSSHConnectOptions", Dict[str, str]]]:
        if isinstance(value, dict):
            # Validate that the dictionary contains only string keys and values
            for key, val in value.items():
                if not isinstance(key, str) or not isinstance(val, str):
                    raise ValueError("All keys and values in connect_options dictionary must be strings.")
        return value

    @field_validator("worker_class")
    def validate_worker_class(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ALLOWED_WORKER_CLASSES:
            raise ValueError(f"worker_class must be one of {ALLOWED_WORKER_CLASSES}")
        return value


class DaskClusterConfig(BaseModel):
    scheduler: SchedulerConfig
    worker: WorkerConfig

# === Note this for configuration settings to work
# The `--preload` option in Dask allows you to specify a module or script that should be loaded by each worker process when it starts. This can be useful for initializing custom logic or setting up the environment for the workers.

# Here is the relevant excerpt from the `distributed/cli/dask_worker.py` file:

# ```python


# @click.option(
#     "--preload",
#     type=str,
#     multiple=True,
#     is_eager=True,
#     help="Module that should be loaded by each worker process "
#     'like "foo.bar" or "/path/to/foo.py"',
# )
# ```

# When you use the `--preload` option, you can specify one or more modules or scripts that will be executed when the worker starts. For example:

# ```bash
# dask-worker SCHEDULER_ADDRESS: 8786 - -preload my_module
# ```

# In this case, `my_module` will be loaded by each worker. You can also specify multiple modules or scripts:

# ```bash
# dask-worker SCHEDULER_ADDRESS: 8786 - -preload module1 - -preload module2
# ```

# This can be particularly useful for running custom initialization code, loading configurations, or setting up logging.

# ===============
# Example usage
if __name__ == "__main__":
    try:
        scheduler_config = SchedulerConfig(
            address="localhost",
            connect_options={"username": "user", "password": "pass"},
            remote_python="/usr/bin/python3",
            scheduler_timeout="120s",
            kwargs=SchedulerCLIOptions(
                host="localhost",
                port=8786,
                dashboard_address=":8787",
                dashboard=True,
                preload="my_module"
            )
        )
        worker_config = WorkerConfig(
            scheduler="localhost",
            address="worker-host",
            worker_class="CustomWorker",
            connect_options={"username": "worker_user", "password": "worker_pass"},
            remote_python="/usr/bin/python3",
            kwargs=WorkerCLIOptions(
                host="worker-host",
                worker_port="3000:3026",
                nanny_port="4000:4026",
                memory_limit="5GB",
                nthreads=2
            )
        )
        cluster_config = DaskClusterConfig(
            scheduler=scheduler_config,
            worker=worker_config
        )
        print(cluster_config.model_dump())
    except ValueError as e:
        print(f"Configuration error: {e}")
