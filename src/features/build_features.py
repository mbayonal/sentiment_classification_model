#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to build features from preprocessed IMDB dataset files.
Works with sampled data to keep file sizes under 100 MB.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """
    Load a TSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the data
    """
    logging.info(f"Loading {file_path}")
    try:
        return pd.read_csv(file_path, sep='\t')
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise

def merge_title_data():
    """
    Merge title-related data into a single DataFrame.
    
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
    # Load the necessary files
    basics_path = Path("data/processed/title.basics.tsv")
    ratings_path = Path("data/processed/title.ratings.tsv")
    
    # Check if files exist
    if not basics_path.exists() or not ratings_path.exists():
        logging.error("Required files don't exist. Please run preprocessing first.")
        return None
    
    # Load the data
    basics_df = load_data(basics_path)
    ratings_df = load_data(ratings_path)
    
    logging.info(f"Basics data shape: {basics_df.shape}")
    logging.info(f"Ratings data shape: {ratings_df.shape}")
    
    # Merge the data with validation
    merged_df = pd.merge(basics_df, ratings_df, on='tconst', how='left', validate="1:1")
    
    logging.info(f"Merged data shape: {merged_df.shape}")
    
    return merged_df

def extract_movie_features(df):
    """
    Extract features for movies only.
    
    Args:
        df (pandas.DataFrame): DataFrame containing merged title data
        
    Returns:
        pandas.DataFrame: DataFrame with extracted features
    """
    # Filter for movies only
    movies_df = df[df['titleType'] == 'movie'].copy()
    logging.info(f"Number of movies: {len(movies_df)}")
    
    # Handle missing values
    movies_df['startYear'] = pd.to_numeric(movies_df['startYear'], errors='coerce')
    
    # Create decade feature (handle NaN values)
    movies_df['decade'] = movies_df['startYear'].apply(
        lambda x: (x // 10) * 10 if not pd.isna(x) else pd.NA
    )
    
    # Create runtime categories
    bins = [0, 60, 90, 120, 180, float('inf')]
    labels = ['Short (<60m)', 'Standard (60-90m)', 'Standard (90-120m)', 'Long (120-180m)', 'Very Long (>180m)']
    movies_df['runtime_category'] = pd.cut(movies_df['runtimeMinutes'], bins=bins, labels=labels)
    
    # Create rating categories (handle NaN values)
    rating_bins = [0, 4, 6, 8, 10]
    rating_labels = ['Poor', 'Average', 'Good', 'Excellent']
    movies_df['rating_category'] = pd.cut(movies_df['averageRating'], bins=rating_bins, labels=rating_labels)
    
    # Create popularity categories based on number of votes
    vote_bins = [0, 1000, 10000, 100000, float('inf')]
    vote_labels = ['Very Low', 'Low', 'Medium', 'High']
    movies_df['popularity'] = pd.cut(movies_df['numVotes'], bins=vote_bins, labels=vote_labels)
    
    # Calculate statistics for sampled data
    logging.info(f"Decade distribution: {movies_df['decade'].value_counts().sort_index()}")
    logging.info(f"Runtime categories: {movies_df['runtime_category'].value_counts()}")
    logging.info(f"Rating categories: {movies_df['rating_category'].value_counts()}")
    logging.info(f"Popularity categories: {movies_df['popularity'].value_counts()}")
    
    return movies_df

def main():
    """
    Main function to build features from preprocessed IMDB dataset files.
    """
    # Create features directory if it doesn't exist
    features_dir = Path("data/processed/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if output file already exists
    output_path = features_dir / "movie_features.csv"
    if output_path.exists():
        logging.info(f"Features file {output_path} already exists. Skipping feature extraction.")
        return
    
    # Merge title data
    merged_df = merge_title_data()
    if merged_df is None:
        return
    
    # Extract movie features
    movies_df = extract_movie_features(merged_df)
    
    # Save the features
    logging.info(f"Saving features to {output_path}")
    movies_df.to_csv(output_path, index=False)
    logging.info(f"Saved {output_path}")
    
    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"Features file size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
