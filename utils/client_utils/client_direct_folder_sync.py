from distributed import Client
from distributed.worker import Worker
from distributed.diagnostics.plugin import WorkerPlugin, UploadDirectory
import os

class UploadAndSetCWDPlugin(WorkerPlugin):
    """
    A plugin that uploads a local directory to each worker and then sets
    it as the current working directory.
    """

    def __init__(self, local_path: str = None, folder_name: str = None):
        """
        local_path: the path for the client machine you intend to upload.
                    Defaults to os.getcwd() if not specified.
        folder_name: optional folder name to store files under worker.local_directory.
                     Ignored if local_path is not specified (i.e. default is used).
        """
        if local_path is None:
            local_path = os.getcwd()
            self.folder_name = None
        else:
            self.folder_name = folder_name or os.path.basename(local_path)
        self.local_path = local_path
        self._uploader = UploadDirectory(local_path, restart=False, update_path=True)
        self.idempotent = False  # This plugin is idempotent
        self.name = "Upload and Set CWD Plugin"  # Name of the plugin

    def setup(self, worker: Worker):
        # Upload the directory using the built-in UploadDirectory plugin
        self._uploader.setup(worker)

        if self.folder_name:
            # Change the worker's current directory to the uploaded folder
            target_dir = os.path.join(worker.local_directory, self.folder_name)
            os.makedirs(target_dir, exist_ok=True)
            os.chdir(target_dir)
        else:
            # If no folder_name is provided, set the working directory to the worker's local_directory
            os.chdir(worker.local_directory)

if __name__ == "__main__":
    client = Client("tcp://localhost:8786")
    plugin = UploadAndSetCWDPlugin()  # Will use current working directory by default
    client.register_plugin(plugin)
    # ... perform work ...
    client.unregister_worker_plugin(plugin.name)  # Unregister when done
