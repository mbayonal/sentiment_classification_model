#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to build features from preprocessed IMDB dataset files.
"""

import os
import pandas as pd
from pathlib import Path

def load_data(file_path):
    """
    Load a TSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the data
    """
    print(f"Loading {file_path}")
    return pd.read_csv(file_path, sep='\t')

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
        print("Required files don't exist. Please run preprocessing first.")
        return None
    
    # Load the data
    basics_df = load_data(basics_path)
    ratings_df = load_data(ratings_path)
    
    # Merge the data
    merged_df = pd.merge(basics_df, ratings_df, on='tconst', how='left')
    
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
    
    # Create decade feature
    movies_df['decade'] = (movies_df['startYear'] // 10) * 10
    
    # Create runtime categories
    bins = [0, 60, 90, 120, 180, float('inf')]
    labels = ['Short (<60m)', 'Standard (60-90m)', 'Standard (90-120m)', 'Long (120-180m)', 'Very Long (>180m)']
    movies_df['runtime_category'] = pd.cut(movies_df['runtimeMinutes'], bins=bins, labels=labels)
    
    # Create rating categories
    rating_bins = [0, 4, 6, 8, 10]
    rating_labels = ['Poor', 'Average', 'Good', 'Excellent']
    movies_df['rating_category'] = pd.cut(movies_df['averageRating'], bins=rating_bins, labels=rating_labels)
    
    # Create popularity categories based on number of votes
    vote_bins = [0, 1000, 10000, 100000, float('inf')]
    vote_labels = ['Very Low', 'Low', 'Medium', 'High']
    movies_df['popularity'] = pd.cut(movies_df['numVotes'], bins=vote_bins, labels=vote_labels)
    
    return movies_df

def main():
    """
    Main function to build features from preprocessed IMDB dataset files.
    """
    # Create features directory if it doesn't exist
    features_dir = Path("data/processed/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge title data
    merged_df = merge_title_data()
    if merged_df is None:
        return
    
    # Extract movie features
    movies_df = extract_movie_features(merged_df)
    
    # Save the features
    output_path = features_dir / "movie_features.csv"
    print(f"Saving features to {output_path}")
    movies_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
