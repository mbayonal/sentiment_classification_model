#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and sample IMDB dataset files to keep them under 100 MB.
"""

import os
import urllib.request
import gzip
import shutil
import pandas as pd
import random
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants for file names
FILE_TITLE_AKAS = "title.akas.tsv.gz"
FILE_TITLE_BASICS = "title.basics.tsv.gz"
FILE_TITLE_CREW = "title.crew.tsv.gz"
FILE_TITLE_EPISODE = "title.episode.tsv.gz"
FILE_TITLE_PRINCIPALS = "title.principals.tsv.gz"
FILE_TITLE_RATINGS = "title.ratings.tsv.gz"
FILE_NAME_BASICS = "name.basics.tsv.gz"

# Base URL for IMDB dataset
BASE_URL = "https://datasets.imdbws.com/"

# List of files to download
FILES = [
    FILE_TITLE_AKAS,
    FILE_TITLE_BASICS,
    FILE_TITLE_CREW,
    FILE_TITLE_EPISODE,
    FILE_TITLE_PRINCIPALS,
    FILE_TITLE_RATINGS,
    FILE_NAME_BASICS
]

# Load parameters from params.yaml
def load_params():
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        target_size_mb = params.get('TARGET_SIZE_MB', 100)
        sampling_ratios = params.get('SAMPLING_RATIOS', {})
        
        # Set default sampling ratios if not specified in params.yaml
        default_ratios = {
            FILE_TITLE_AKAS: 0.05,      # 5% of original
            FILE_TITLE_BASICS: 0.1,    # 10% of original
            FILE_TITLE_CREW: 0.1,      # 10% of original
            FILE_TITLE_EPISODE: 0.1,   # 10% of original
            FILE_TITLE_PRINCIPALS: 0.05, # 5% of original
            FILE_TITLE_RATINGS: 0.2,   # 20% of original
            FILE_NAME_BASICS: 0.05     # 5% of original
        }
        
        # Use default ratios for any missing files
        for file, ratio in default_ratios.items():
            if file not in sampling_ratios:
                sampling_ratios[file] = ratio
        
        return target_size_mb, sampling_ratios
    
    except Exception as e:
        logging.warning(f"Error loading params.yaml: {e}. Using default values.")
        return 100, {
            FILE_TITLE_AKAS: 0.05,
            FILE_TITLE_BASICS: 0.1,
            FILE_TITLE_CREW: 0.1,
            FILE_TITLE_EPISODE: 0.1,
            FILE_TITLE_PRINCIPALS: 0.05,
            FILE_TITLE_RATINGS: 0.2,
            FILE_NAME_BASICS: 0.05
        }

# Load parameters
TARGET_SIZE_MB, SAMPLING_RATIOS = load_params()
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

logging.info(f"Target size: {TARGET_SIZE_MB} MB")
logging.info(f"Sampling ratios: {SAMPLING_RATIOS}")


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
    
    # Get file size
    file_size = output_path.stat().st_size
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")
    
    return file_size

def sample_gz_tsv(input_path, output_path, sample_ratio):
    """
    Sample a gzipped TSV file to reduce its size.
    
    Args:
        input_path (Path): Path to the input gzipped TSV file
        output_path (Path): Path to save the sampled gzipped TSV file
        sample_ratio (float): Ratio of rows to keep (0.0 to 1.0)
    """
    print(f"Sampling {input_path} with ratio {sample_ratio}")
    
    # Create a temporary file for the uncompressed data
    temp_path = input_path.with_suffix('')
    
    # Decompress the file
    with gzip.open(input_path, 'rb') as f_in:
        with open(temp_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Read the header and a sample of rows
    with open(temp_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        lines = f.readlines()
    
    # Sample the lines
    sampled_lines = random.sample(lines, int(len(lines) * sample_ratio))
    
    # Write the sampled data to a temporary file
    sampled_temp_path = temp_path.with_suffix('.sampled')
    with open(sampled_temp_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.writelines(sampled_lines)
    
    # Compress the sampled data
    with open(sampled_temp_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up temporary files
    temp_path.unlink()
    sampled_temp_path.unlink()
    
    # Get file size
    file_size = output_path.stat().st_size
    print(f"Sampled file size: {file_size / (1024 * 1024):.2f} MB")
    
    return file_size

def main():
    """
    Main function to download and sample IMDB dataset files.
    """
    # Create raw data directory if it doesn't exist
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a directory for temporary files
    temp_dir = raw_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Download and sample each file
    for file in FILES:
        url = BASE_URL + file
        temp_path = temp_dir / file
        output_path = raw_dir / file
        
        # Skip if file already exists
        if output_path.exists():
            print(f"File {output_path} already exists. Skipping download.")
            continue
        
        # Download the file to a temporary location
        file_size = download_file(url, temp_path)
        
        # Check if the file needs sampling
        if file_size > TARGET_SIZE_BYTES:
            print(f"File {file} is larger than {TARGET_SIZE_MB} MB. Sampling...")
            sample_ratio = SAMPLING_RATIOS.get(file, 0.1)  # Default to 10% if not specified
            sample_gz_tsv(temp_path, output_path, sample_ratio)
            # Remove the original file
            temp_path.unlink()
        else:
            # Just move the file if it's already small enough
            temp_path.rename(output_path)
            print(f"File {file} is already smaller than {TARGET_SIZE_MB} MB. No sampling needed.")
    
    # Clean up the temporary directory
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            print(f"Could not remove temporary directory {temp_dir}. It may not be empty.")

if __name__ == "__main__":
    main()
