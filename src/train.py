import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
import mlflow
import mlflow.sklearn
import optuna
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='data/processed/processed.csv'):
    """Load processed data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    return pd.read_csv(file_path)

def objective(trial, X_train, y_train):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    return f1_score(y_train, y_pred)

def train_model():
    """Train RandomForest model with validation and cross-validation."""
    # Load data
    df = load_data()
    features = ['Recency', 'Frequency', 'Monetary']
    X = df[features]
    y = df['is_high_risk']
    
    # Train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.36, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5556, random_state=42)
    
    # Print class distributions
    print(f"Train: {np.bincount(y_train)}")
    print(f"Validation: {np.bincount(y_val)}")
    print(f"Test: {np.bincount(y_test)}")
    
    # Optuna hyperparameter tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    
    # Train model with best parameters
    model = RandomForestClassifier(**best_params, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        model.fit(X_train, y_train)
        
        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        print(f"Validation ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
        
        # Log metrics and model
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        
        # Cross-validation to check overfitting
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("Model saved to models/random_forest_model.pkl")
    
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    # Test set evaluation
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_test_proba):.4f}, F1: {f1_score(y_test, y_test_pred):.4f}")