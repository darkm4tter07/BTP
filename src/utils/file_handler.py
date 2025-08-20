import os
import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_paths():
    """Create necessary directories."""
    config = load_config()
    base_path = Path(config['paths']['data_root'])
    
    directories = [
        'datasets/construction_videos',
        'datasets/annotated_data',
        'training_outputs/activity_models',
        'training_outputs/pose_models',
        'results/inference_outputs'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")

def get_drive_path(relative_path):
    """Get full Google Drive path."""
    config = load_config()
    return os.path.join(config['paths']['data_root'], relative_path)