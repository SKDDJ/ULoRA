from datasets import load_dataset
import os

# Get the current file path and append the subpath
current_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

dataset = load_dataset(
    path="Shiym/SD-FineTune",    # Dataset ID or local path
    cache_dir=current_file_path,  # Cache in the specified subpath
    token=None,                   # HuggingFace token for private datasets
    trust_remote_code=True,       # Trust (or not) the remote code
)
