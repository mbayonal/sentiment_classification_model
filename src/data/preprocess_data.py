#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to preprocess sampled IMDB dataset files.
"""

import os
import pandas as pd
import gzip
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    logging.info(f"Reading {file_path}")
    try:
        # Try with low_memory=False first
        return pd.read_csv(file_path, sep='\t', low_memory=False)
    except Exception as e:
        logging.warning(f"Error reading {file_path} with low_memory=False: {e}")
        logging.info("Trying again with low_memory=True and chunksize")
        
        # Try reading in chunks
        chunks = pd.read_csv(file_path, sep='\t', low_memory=True, chunksize=100000)
        return pd.concat(chunks)

def preprocess_title_basics(df):
    """
    Preprocess title.basics.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.basics data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    logging.info("Preprocessing title.basics data")
    
    # Convert startYear and endYear to numeric, handling '\N' values
    df['startYear'] = pd.to_numeric(df['startYear'].replace('\\N', pd.NA), errors='coerce')
    df['endYear'] = pd.to_numeric(df['endYear'].replace('\\N', pd.NA), errors='coerce')
    
    # Convert runtimeMinutes to numeric
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'].replace('\\N', pd.NA), errors='coerce')
    
    # Convert isAdult to boolean
    df['isAdult'] = df['isAdult'].astype(bool)
    
    # Split genres into a list
    df['genres'] = df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    
    return df

def preprocess_title_ratings(df):
    """
    Preprocess title.ratings.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.ratings data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    logging.info("Preprocessing title.ratings data")
    
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
    logging.info("Preprocessing name.basics data")
    
    # Convert birthYear and deathYear to numeric, handling '\N' values
    df['birthYear'] = pd.to_numeric(df['birthYear'].replace('\\N', pd.NA), errors='coerce')
    df['deathYear'] = pd.to_numeric(df['deathYear'].replace('\\N', pd.NA), errors='coerce')
    
    # Split primaryProfession and knownForTitles into lists
    df['primaryProfession'] = df['primaryProfession'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    df['knownForTitles'] = df['knownForTitles'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    
    return df

def preprocess_title_akas(df):
    """
    Preprocess title.akas.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.akas data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    logging.info("Preprocessing title.akas data")
    
    # Convert ordering to numeric
    df['ordering'] = pd.to_numeric(df['ordering'], errors='coerce')
    
    # Convert isOriginalTitle to boolean
    df['isOriginalTitle'] = df['isOriginalTitle'].astype(bool)
    
    # Handle types and attributes arrays
    df['types'] = df['types'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    df['attributes'] = df['attributes'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    
    return df

def preprocess_title_crew(df):
    """
    Preprocess title.crew.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.crew data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    logging.info("Preprocessing title.crew data")
    
    # Split directors and writers into lists
    df['directors'] = df['directors'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    df['writers'] = df['writers'].apply(lambda x: x.split(',') if isinstance(x, str) and x != '\\N' else [])
    
    return df

def preprocess_title_episode(df):
    """
    Preprocess title.episode.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.episode data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    logging.info("Preprocessing title.episode data")
    
    # Convert seasonNumber and episodeNumber to numeric
    df['seasonNumber'] = pd.to_numeric(df['seasonNumber'].replace('\\N', pd.NA), errors='coerce')
    df['episodeNumber'] = pd.to_numeric(df['episodeNumber'].replace('\\N', pd.NA), errors='coerce')
    
    return df

def preprocess_title_principals(df):
    """
    Preprocess title.principals.tsv.gz file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing title.principals data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    logging.info("Preprocessing title.principals data")
    
    # Convert ordering to numeric
    df['ordering'] = pd.to_numeric(df['ordering'], errors='coerce')
    
    # Handle characters field (might be JSON-like string)
    df['characters'] = df['characters'].apply(lambda x: x if isinstance(x, str) and x != '\\N' else None)
    
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
            logging.warning(f"File {input_path} doesn't exist. Skipping preprocessing.")
            continue
        
        # Skip if output file already exists
        if output_path.exists():
            logging.info(f"File {output_path} already exists. Skipping preprocessing.")
            continue
        
        try:
            # Read the file
            df = read_gz_tsv(input_path)
            
            # Apply specific preprocessing based on the file
            if 'title.basics' in file:
                df = preprocess_title_basics(df)
            elif 'title.ratings' in file:
                df = preprocess_title_ratings(df)
            elif 'name.basics' in file:
                df = preprocess_name_basics(df)
            elif 'title.akas' in file:
                df = preprocess_title_akas(df)
            elif 'title.crew' in file:
                df = preprocess_title_crew(df)
            elif 'title.episode' in file:
                df = preprocess_title_episode(df)
            elif 'title.principals' in file:
                df = preprocess_title_principals(df)
            
            # Save the preprocessed file
            logging.info(f"Saving preprocessed data to {output_path}")
            df.to_csv(output_path, sep='\t', index=False)
            logging.info(f"Saved {output_path}")
            
        except Exception as e:
            logging.error(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
