from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json
import yaml
import toml
from models.data_models.dask_config_models import DaskClusterConfig

class DaskClusterSettings(BaseSettings):
    # This setting holds our entire cluster configuration:
    cluster_config: DaskClusterConfig

    # You can also load environment variables from an .env file, if needed.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def from_file(cls, file_path: Path) -> "DaskClusterSettings":
        """
        Load configuration from a JSON, YAML, or TOML file.
        Only syntax validation is performed; this does not check for file existence
        on remote systems.
        """
        ext = file_path.suffix.lower()
        with file_path.open("r") as f:
            if ext == ".json":
                data = json.load(f)
            elif ext in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif ext == ".toml":
                data = toml.load(f)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        return cls(cluster_config=DaskClusterConfig(**data))


# ...existing code...
if __name__ == "__main__":
    # Change the file path to our scheduler YAML configuration
    config_file = Path("settings/cluster_config/scheduler.yaml")
    try:
        settings = DaskClusterSettings.from_file(config_file)
        print("Loaded DaskClusterConfig:")
        print(settings.cluster_config.model_dump())
    except Exception as e:
        print(f"Error loading settings: {e}")
