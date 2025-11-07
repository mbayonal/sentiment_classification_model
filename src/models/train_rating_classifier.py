#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train rating category classification models on IMDb movie features.

Predicts rating_category (Poor, Average, Good, Excellent) from movie features:
- decade, runtimeMinutes, genres, popularity, etc.

Models:
- Logistic Regression (multiclass)
- Linear SVM (multiclass)

All experiments logged to MLflow with parameters, metrics, and artifacts.
Best model saved to models/best_model.pkl for deployment.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "features" / "movie_features.csv"
MODELS_DIR = ROOT / "models"
PARAMS_PATH = ROOT / "params.yaml"


def load_params():
    """Load parameters from params.yaml"""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('rating_classifier', {})
    return {}


def load_and_prepare_data():
    """Load movie features and prepare for training"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}\nRun: dvc repro build_features")
    
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded {len(df)} movies")
    
    # Filter movies with valid rating_category
    df = df[df['rating_category'].notna()].copy()
    logging.info(f"Movies with rating: {len(df)}")
    logging.info(f"Rating distribution:\n{df['rating_category'].value_counts()}")
    
    # Select features for training
    # Numeric features
    numeric_features = ['startYear', 'runtimeMinutes', 'numVotes', 'averageRating']
    
    # Categorical features
    categorical_features = ['runtime_category', 'popularity']
    
    # Target
    y = df['rating_category'].values
    
    # Prepare feature dataframe
    X = df[numeric_features + categorical_features].copy()
    
    # Handle missing values
    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())
    
    for col in categorical_features:
        X[col] = X[col].fillna('Unknown')
    
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Features: {list(X.columns)}")
    
    return X, y, numeric_features, categorical_features


def create_models(params, numeric_features, categorical_features):
    """Create model pipelines with preprocessing"""
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    models = {
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(
                C=params.get('logistic_regression', {}).get('C', 1.0),
                max_iter=params.get('logistic_regression', {}).get('max_iter', 1000),
                random_state=42,
                multi_class='multinomial'
            ))
        ]),
        'linear_svm': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LinearSVC(
                C=params.get('linear_svm', {}).get('C', 1.0),
                max_iter=params.get('linear_svm', {}).get('max_iter', 2000),
                random_state=42
            ))
        ])
    }
    
    return models


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_score_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro')
    }
    report = classification_report(y_test, y_pred, output_dict=True)
    return metrics, report


def train_and_log(name, model, X_train, X_test, y_train, y_test, params):
    """Train model and log to MLflow"""
    with mlflow.start_run(run_name=name):
        logging.info(f"\nTraining: {name}")
        
        # Log parameters
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        for key, value in params.get(name, {}).items():
            mlflow.log_param(f"model_{key}", value)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics, report = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logging.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Save report
        report_path = MODELS_DIR / f"{name}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))
        
        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        return metrics, model


def main():
    logging.info("="*60)
    logging.info("IMDb Rating Category Classification Training")
    logging.info("="*60)
    
    # MLflow experiment
    mlflow.set_experiment("imdb-rating-classification")
    
    # Load parameters
    params = load_params()
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    
    # Load data
    X, y, numeric_features, categorical_features = load_and_prepare_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create and train models
    models = create_models(params, numeric_features, categorical_features)
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        metrics, trained_model = train_and_log(name, model, X_train, X_test, y_train, y_test, params)
        results[name] = metrics
        trained_models[name] = trained_model
    
    # Select best model based on F1 score (weighted)
    best_name = max(results, key=lambda k: results[k]['f1_score_weighted'])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]
    
    logging.info("\n" + "="*60)
    logging.info(f"BEST MODEL: {best_name}")
    logging.info(f"F1 Score (weighted): {best_metrics['f1_score_weighted']:.4f}")
    logging.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
    logging.info("="*60)
    
    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Saved best model to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_name,
        'metrics': best_metrics,
        'parameters': params,
        'target_classes': ['Poor', 'Average', 'Good', 'Excellent']
    }
    metadata_path = MODELS_DIR / "best_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
