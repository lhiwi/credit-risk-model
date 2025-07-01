import pandas as pd
import os
from src.data_processing import (
    RFMTransformer,
    TemporalAggregator,
    CategoricalAggregator,
    build_and_run_pipeline,
)

# Sample mini-dataset
SAMPLE = pd.DataFrame(
    [
        {
            "CustomerId": "C1",
            "TransactionId": "T1",
            "TransactionStartTime": "2025-01-01",
            "Value": 100,
            "ProductCategory": "A",
            "ChannelId": "X",
            "PricingStrategy": 1,
        },
        {
            "CustomerId": "C1",
            "TransactionId": "T2",
            "TransactionStartTime": "2025-01-05",
            "Value": 200,
            "ProductCategory": "B",
            "ChannelId": "X",
            "PricingStrategy": 1,
        },
        {
            "CustomerId": "C2",
            "TransactionId": "T3",
            "TransactionStartTime": "2025-01-03",
            "Value": 150,
            "ProductCategory": "A",
            "ChannelId": "Y",
            "PricingStrategy": 2,
        },
    ],
)


def test_rfm_transformer():
    rfm = RFMTransformer().transform(SAMPLE)
    assert set(rfm.columns) == {"CustomerId", "Recency", "Frequency", "Monetary"}
    # C1 frequency = 2, C2 = 1
    assert rfm.set_index("CustomerId")["Frequency"].to_dict() == {"C1": 2, "C2": 1}


def test_temporal_aggregator():
    temp = TemporalAggregator().transform(SAMPLE)
    assert set(temp.columns) == {"CustomerId", "hour", "day", "month"}


def test_categorical_aggregator():
    cat = CategoricalAggregator(
        ["ProductCategory", "ChannelId", "PricingStrategy"]
    ).transform(SAMPLE)
    # Expect columns like 'ProductCategory_A', 'ChannelId_X', 'PricingStrategy_1'
    assert "ProductCategory_A" in cat.columns
    assert "ChannelId_X" in cat.columns


def test_full_pipeline(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "features.csv"
    SAMPLE.to_csv(raw, index=False)
    build_and_run_pipeline(str(raw), str(out))
    assert out.exists()
    df = pd.read_csv(out)
    # Should have CustomerId plus at least 3+3 columns
    assert "CustomerId" in df.columns
    assert "Recency" in df.columns
    assert df.CustomerId.nunique() == 2
