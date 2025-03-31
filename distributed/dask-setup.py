#%% Setting up the configurations
import dask.config
import pprint
import dask
cluster_config = {
    "0.0.0.0" : {"connect_options": {},
                 "scheduler_options": {},
                 },
}


#%% Setting up the cluster and spinning up the workers for making ready using SSH Cluster
from dask.distributed import Client, SSHCluster
cluster = SSHCluster(
    ["0.0.0.0", "Workstation", "Mobilework","Desktop"],
    connect_options={"known_hosts": None}, # Can be a list corresponding to each host
    worker_options={"nprocs": 4, "nthreads": 4,
                    "resources": {"GPU": 0}, "worker_class": "dask_cuda.CUDAWorker"},
    # scheduler_options={"dashboard_address": ":8786", "port": 8786}, # Can be configured with number of other workers tool
    # scheduler_options={"port": 0, "dashboard_address": ":8797"},
    worker_class="dask_cuda.CUDAWorker") # We have to be very careful here
client = Client(cluster)
#%%

#%% Setting up the scheduler and workers using CLI Manually
# Setup the scheduler first and then run the workers on the corresponding nodes

# Step 1 : Run the scheduler
# dask scheduler --host "localhost" --port 8786 # Use default ports always if possible
# Can add the following --jupyter --show 

# Step 2 : Spin up the workers on the nodes
# dask worker --nanny-port 8001 --dashboard-address 8000 --name --resources"GPU=0" --worker-class "dask_cuda.CUDAWorker" --nprocs 1 --nthreads 1 --memory-limit 0 --no-bokeh --no-nanny --local-directory /tmp --name "worker-1" --resources "GPU=0" --worker-class "dask_cuda.CUDAWorker" --nprocs 1 --nthreads 1 --memory-limit 0 --no-bokeh --no-nanny 
# dask worker --dashboard-address 8000 --name Worker-desktop --resources "GPU=0" --worker-class "dask_cuda.CUDAWorker" --host "Workstation"


# To allow ports

# sudo netstat - tulnp | grep 8786

# CUDA_VISIBLE_DEVICES=0 dask-cuda-worker tcp://127.0.0.1:8786 --name Workstation-main

# dask scheduler --host "localhost" --port 8786 --jupyter --show # To listen locally only
# dask scheduler --host "0.0.0.0" --port 8786 --jupyter --show # To listen on LAN
#%% Initializing the client using a simple local Client - with already started scheduler
from dask.distributed import Client
client = Client("tcp:localhost:8786")
print(client)
#%% Setting up a local CUDA Cluster
from dask.distributed import LocalCluster
cluster = LocalCluster(processes=True, worker_class="dask_cuda.CUDAWorker")
print(cluster)
client = Client(cluster)
print(client)
# %% For Debugging - Single Threaded scheduler
# overwrite default with single-threaded scheduler
import dask
from pprint import pprint
dask.config.set(scheduler='synchronous')

pprint(dask.config.config)

#%%