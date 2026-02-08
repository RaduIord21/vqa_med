#!/usr/bin/env python3
"""Download Medical VQA ImageCLEF 2019 dataset from Kaggle.

Usage on Google Colab:
    1. Upload kaggle.json to .kaggle/kaggle.json
    2. Run: python download_dataset.py
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_kaggle_auth():
    """Setup Kaggle authentication."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json_dest = kaggle_dir / "kaggle.json"
    
    # Check for kaggle.json in current directory or .kaggle subdirectory
    local_kaggle_paths = [
        Path("kaggle.json"),
        Path(".kaggle/kaggle.json"),
    ]
    
    for local_path in local_kaggle_paths:
        if local_path.exists():
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(local_path, kaggle_json_dest)
            os.chmod(kaggle_json_dest, 0o600)
            print(f"Kaggle credentials copied from {local_path}")
            return True
    
    if kaggle_json_dest.exists():
        print("Kaggle credentials found in ~/.kaggle/kaggle.json")
        return True
    
    print("ERROR: kaggle.json not found!")
    print("Place your kaggle.json in .kaggle/kaggle.json")
    return False


def install_kaggle():
    """Install kaggle package if not available."""
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])


def download_dataset():
    """Download and extract the Medical VQA dataset."""
    dataset_name = "claudiopisa9884/medical-vqa-imageclef-2019"
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading: {dataset_name}")
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=str(data_dir), unzip=True)
    
    print("Download complete!")
    print("\nContents:")
    for item in sorted(data_dir.iterdir()):
        print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")


def main():
    install_kaggle()
    if not setup_kaggle_auth():
        sys.exit(1)
    download_dataset()
    print("\nReady! Run: python train.py")


if __name__ == "__main__":
    main()
