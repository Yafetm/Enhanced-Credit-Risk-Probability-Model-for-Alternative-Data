import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
from src.data_processing import calculate_rfm, create_proxy
import pytz

def test_rfm_calc():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': ['T1', 'T2', 'T3'],
        'TransactionStartTime': ['2018-11-01T00:00:00Z', '2018-11-02T00:00:00Z', '2018-11-01T00:00:00Z'],
        'Amount': [100, 200, 150]
    })
    rfm = calculate_rfm(df)
    assert rfm.shape[0] == 2
    assert rfm.loc[1, 'Monetary'] == 300
    assert rfm.loc[2, 'Frequency'] == 1
    assert rfm.loc[1, 'Recency'] == (pd.to_datetime('2019-02-01').tz_localize('UTC') - pd.to_datetime('2018-11-02T00:00:00Z')).days

def test_create_proxy():
    rfm = pd.DataFrame({
        'Recency': [10, 20],
        'Frequency': [1, 10],
        'Monetary': [100, 1000]
    }, index=[1, 2])
    rfm_with_proxy = create_proxy(rfm)
    assert 'is_high_risk' in rfm_with_proxy.columns
    assert rfm_with_proxy['is_high_risk'].isin([0, 1]).all()