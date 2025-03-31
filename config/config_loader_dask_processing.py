import os
import dask.config

def load_config(file_path: str) -> dict:
    """
    Load Dask configuration settings from a JSON or TOML file.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        dict: The configuration dictionary that was loaded.
    
    Raises:
        ValueError: If the file format is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    config_data = {}
    if ext == '.toml':
        try:
            # For Python 3.11+ tomllib is available in the standard library.
            import tomllib
        except ImportError:
            # Otherwise, fallback to the third-party 'toml' library.
            import toml as tomllib
        with open(file_path, 'rb') as f:
            config_data = tomllib.load(f)
    elif ext == '.json':
        import json
        with open(file_path, 'r') as f:
            config_data = json.load(f)
    else:
        raise ValueError("Unsupported configuration file format. Supported formats are .toml and .json")
    
    # Update the current Dask configuration with the loaded settings.
    dask.config.update(config_data)
    return config_data

if __name__ == '__main__':
    import pprint
    # Make sure the configuration file is in the same directory first.
    config_file = 'config.toml'  # Change this path if necessary
    cfg = load_config(config_file)
    pprint.pprint(cfg)