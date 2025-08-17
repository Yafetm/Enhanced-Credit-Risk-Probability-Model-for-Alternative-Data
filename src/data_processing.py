import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import yaml
import pickle
import pytz

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def calculate_rfm(df):
    snapshot_date = pd.to_datetime(config['rfm_snapshot_date']).tz_localize('UTC')
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    )
    print("RFM DataFrame shape:", rfm.shape)
    print("RFM Sample:", rfm.head())
    return rfm

def create_proxy(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    freq_threshold = rfm['Frequency'].quantile(0.2)
    monetary_threshold = rfm['Monetary'].quantile(0.2)
    rfm['is_high_risk'] = ((rfm['Frequency'] <= freq_threshold) & (rfm['Monetary'] <= monetary_threshold)).astype(int)
    print("is_high_risk counts:", rfm['is_high_risk'].value_counts())
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return rfm

def preprocess_features(df):
    numerical_cols = ['Amount', 'Value']
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])
    transformed = preprocessor.fit_transform(df)
    transformed_df = pd.DataFrame(
        transformed,
        columns=numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out()),
        index=df.index
    )
    with open('data/processed/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    return transformed_df

if __name__ == "__main__":
    df = pd.read_csv(config['data_path'])
    print("Input data shape:", df.shape)
    processed_features = preprocess_features(df)
    rfm = calculate_rfm(df)
    rfm_with_proxy = create_proxy(rfm)
    processed = rfm_with_proxy.join(df.groupby('CustomerId').first()[['ProductCategory', 'ChannelId', 'PricingStrategy']])
    processed = processed.join(processed_features.groupby(df['CustomerId']).mean())
    processed.to_csv('data/processed/processed.csv', index=True)
    print("Processed data saved. Shape:", processed.shape)