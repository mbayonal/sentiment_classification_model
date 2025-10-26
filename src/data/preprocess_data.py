#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to preprocess IMDB dataset files.
"""

import os
import pandas as pd
import gzip
from pathlib import Path

# List of files to process
FILES = [
    "title.akas.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.ratings.tsv.gz",
    "name.basics.tsv.gz"
]

def read_gz_tsv(file_path):
    """
    Read a gzipped TSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the gzipped TSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the data
    """
    print(f"Reading {file_path}")
    return pd.read_csv(file_path, sep='\t', low_memory=False)

def preprocess_title_basics(df):
    """
    Preprocess title.basics.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.basics data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    # Convert startYear and endYear to numeric, handling '\N' values
    df['startYear'] = pd.to_numeric(df['startYear'].replace('\\N', pd.NA), errors='coerce')
    df['endYear'] = pd.to_numeric(df['endYear'].replace('\\N', pd.NA), errors='coerce')
    
    # Convert runtimeMinutes to numeric
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'].replace('\\N', pd.NA), errors='coerce')
    
    # Convert isAdult to boolean
    df['isAdult'] = df['isAdult'].astype(bool)
    
    # Split genres into a list
    df['genres'] = df['genres'].apply(lambda x: x.split(',') if x != '\\N' else [])
    
    return df

def preprocess_title_ratings(df):
    """
    Preprocess title.ratings.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.ratings data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    # Convert averageRating and numVotes to numeric
    df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
    df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce')
    
    return df

def preprocess_name_basics(df):
    """
    Preprocess name.basics.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing name.basics data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    # Convert birthYear and deathYear to numeric, handling '\N' values
    df['birthYear'] = pd.to_numeric(df['birthYear'].replace('\\N', pd.NA), errors='coerce')
    df['deathYear'] = pd.to_numeric(df['deathYear'].replace('\\N', pd.NA), errors='coerce')
    
    # Split primaryProfession and knownForTitles into lists
    df['primaryProfession'] = df['primaryProfession'].apply(lambda x: x.split(',') if x != '\\N' else [])
    df['knownForTitles'] = df['knownForTitles'].apply(lambda x: x.split(',') if x != '\\N' else [])
    
    return df

def main():
    """
    Main function to preprocess IMDB dataset files.
    """
    # Create processed data directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for file in FILES:
        input_path = Path("data/raw") / file
        output_path = processed_dir / file.replace('.gz', '')
        
        # Skip if input file doesn't exist
        if not input_path.exists():
            print(f"File {input_path} doesn't exist. Skipping preprocessing.")
            continue
        
        # Read the file
        df = read_gz_tsv(input_path)
        
        # Apply specific preprocessing based on the file
        if 'title.basics' in file:
            df = preprocess_title_basics(df)
        elif 'title.ratings' in file:
            df = preprocess_title_ratings(df)
        elif 'name.basics' in file:
            df = preprocess_name_basics(df)
        
        # Save the preprocessed file
        print(f"Saving preprocessed data to {output_path}")
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
