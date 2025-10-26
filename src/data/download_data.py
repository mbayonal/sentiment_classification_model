#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download IMDB dataset files.
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path

# Base URL for IMDB dataset
BASE_URL = "https://datasets.imdbws.com/"

# List of files to download
FILES = [
    "title.akas.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.ratings.tsv.gz",
    "name.basics.tsv.gz"
]

def download_file(url, output_path):
    """
    Download a file from a URL to a specified path.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file to
    """
    print(f"Downloading {url} to {output_path}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded {output_path}")

def main():
    """
    Main function to download IMDB dataset files.
    """
    # Create raw data directory if it doesn't exist
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each file
    for file in FILES:
        url = BASE_URL + file
        output_path = raw_dir / file
        
        # Skip if file already exists
        if output_path.exists():
            print(f"File {output_path} already exists. Skipping download.")
            continue
        
        # Download the file
        download_file(url, output_path)

if __name__ == "__main__":
    main()
