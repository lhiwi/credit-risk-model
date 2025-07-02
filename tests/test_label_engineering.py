import pandas as pd
from src.label_engineering import create_proxy_label

# Sample DataFrame with two customers
SAMPLE = pd.DataFrame([
    # Customer A: high Recency, low F/M → should be high-risk
    {'CustomerId':'A','Recency':30,'Frequency':1,'Monetary':100},
    {'CustomerId':'A','Recency':35,'Frequency':1,'Monetary':150},
    # Customer B: low Recency, high F/M → low-risk
    {'CustomerId':'B','Recency':5,'Frequency':10,'Monetary':2000},
    {'CustomerId':'B','Recency':3,'Frequency':8,'Monetary':1800},
])

def test_create_proxy_label_assigns_high_risk_correctly():
    # First, aggregate sample to one row per customer
    agg = SAMPLE.groupby('CustomerId').agg({
        'Recency':'mean','Frequency':'mean','Monetary':'sum'
    }).reset_index()
    # Create labels
    labeled = create_proxy_label(agg, ['Recency','Frequency','Monetary'], n_clusters=2, random_state=0)
    # The customer with higher Recency (A) must be flagged
    flags = labeled.set_index('CustomerId')['is_high_risk'].to_dict()
    assert flags['A'] == 1
    assert flags['B'] == 0

def test_create_proxy_label_keeps_all_columns():
    agg = SAMPLE.groupby('CustomerId').agg({
        'Recency':'mean','Frequency':'mean','Monetary':'sum'
    }).reset_index()
    labeled = create_proxy_label(agg, ['Recency','Frequency','Monetary'], n_clusters=2, random_state=0)
    # Ensure original columns still present
    for col in ['CustomerId','Recency','Frequency','Monetary','is_high_risk']:
        assert col in labeled.columns
