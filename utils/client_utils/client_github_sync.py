from dask.distributed import Client
import os
import subprocess
from dask.distributed import WorkerPlugin

class RepoSyncPlugin(WorkerPlugin):
    """
    A Dask WorkerPlugin that clones (or updates) a GitHub repository
    on each worker to ensure a consistent workspace folder.
    """

    def __init__(self, repo_url, branch='main', target_dir='workspace'):
        self.repo_url = repo_url
        self.branch = branch
        self.target_dir = target_dir

    def setup(self, worker):
        # Attempt to clone or pull updates for the target repository
        if not os.path.exists(self.target_dir):
            subprocess.check_call(['git', 'clone', '-b', self.branch, self.repo_url, self.target_dir])
        else:
            # If the directory exists, pull latest changes
            cwd = os.getcwd()
            try:
                os.chdir(self.target_dir)
                subprocess.check_call(['git', 'fetch', 'origin', self.branch])
                subprocess.check_call(['git', 'checkout', self.branch])
                subprocess.check_call(['git', 'pull', 'origin', self.branch])
            finally:
                os.chdir(cwd)

if __name__ == "__main__":
    # Example implementation
    client = Client("tcp://scheduler-address:8786")
    plugin = RepoSyncPlugin(
        repo_url="https://github.com/your-username/your-repo.git",
        branch="my-dev-branch",
        target_dir="workspace"
    )
    client.register_worker_plugin(plugin)
