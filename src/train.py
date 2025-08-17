import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import optuna
import yaml
import numpy as np

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load processed data
df = pd.read_csv('data/processed/processed.csv')
# Convert integer columns to float to avoid MLflow schema issues
for col in df.select_dtypes(include=['int64']).columns:
    if col not in ['is_high_risk', 'CustomerId']:
        df[col] = df[col].astype(float)
X = df.drop(['is_high_risk', 'CustomerId', 'ProductCategory', 'ChannelId', 'PricingStrategy'], axis=1)
y = df['is_high_risk']
# Split into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Debug class distribution
print("y_train class distribution:", y_train.value_counts())
print("y_val class distribution:", y_val.value_counts())
print("y_test class distribution:", y_test.value_counts())

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    print(f"Trial ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    return roc_auc

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    mlflow.start_run()
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    # Log model with input example
    input_example = X_train.iloc[:5].astype(float).to_dict(orient='records')
    mlflow.sklearn.log_model(model, name='random_forest', input_example=input_example)
    mlflow.end_run()