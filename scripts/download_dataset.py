#!/usr/bin/env python3
"""
Script to download the LendingClub dataset from Zenodo
"""
import os
import requests
from pathlib import Path

def download_lendingclub_dataset():
    """
    Download the LendingClub dataset from Zenodo
    """
    # Zenodo URL for the LendingClub dataset
    url = "https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv"
    
    # Define the data directory
    data_dir = Path("data")
    raw_data_dir = data_dir / "raw"
    
    # Create directories if they don't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the file path
    file_path = raw_data_dir / "LC_loans_granting_model_dataset.csv"
    
    print(f"Downloading LendingClub dataset from {url}")
    print(f"Saving to {file_path}")
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded_size += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded_size / total_size) * 100
                    print(f"\rDownloaded: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="")
    
    print(f"\nDownload completed successfully! File saved to {file_path}")
    print(f"File size: {file_path.stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    download_lendingclub_dataset()