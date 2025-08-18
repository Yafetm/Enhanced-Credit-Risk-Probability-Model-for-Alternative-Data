import pandas as pd
import joblib
import argparse
import os

def load_model(model_path='models/random_forest_model.pkl'):
    """Load the trained RandomForest model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def predict_risk(input_data, model):
    """Predict risk labels for input data."""
    required_columns = ['Recency', 'Frequency', 'Monetary']
    if not all(col in input_data.columns for col in required_columns):
        raise ValueError(f"Input data must contain columns: {required_columns}")
    
    predictions = model.predict(input_data[required_columns])
    probabilities = model.predict_proba(input_data[required_columns])[:, 1]
    
    input_data['is_high_risk'] = predictions
    input_data['risk_probability'] = probabilities
    return input_data

def main(data_path):
    """Load data, predict, and save results."""
    # Load model
    model = load_model()
    
    # Load input data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    input_data = pd.read_csv(data_path)
    
    # Predict
    result = predict_risk(input_data, model)
    
    # Save predictions
    os.makedirs('data/predictions', exist_ok=True)
    output_path = 'data/predictions/predictions.csv'
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(result[['Recency', 'Frequency', 'Monetary', 'is_high_risk', 'risk_probability']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict credit risk using trained model")
    parser.add_argument('--data_path', type=str, default='data/processed/processed.csv', 
                       help='Path to input data CSV')
    args = parser.parse_args()
    main(args.data_path)