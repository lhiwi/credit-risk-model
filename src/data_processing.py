import argparse
import json
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class RFMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        snap = (
            self.snapshot_date
            or df["TransactionStartTime"].max() + pd.Timedelta(days=1)
        )
        rfm = (
            df.groupby("CustomerId")
            .agg(
                Recency=("TransactionStartTime", lambda ts: (snap - ts.max()).days),
                Frequency=("TransactionId", "count"),
                Monetary=("Value", "sum"),
            )
            .reset_index()
        )
        return rfm[["CustomerId", "Recency", "Frequency", "Monetary"]]


class TemporalAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        ts = pd.to_datetime(df["TransactionStartTime"])
        agg = (
            pd.DataFrame(
                {
                    "CustomerId": df["CustomerId"],
                    "hour": ts.dt.hour,
                    "day": ts.dt.day,
                    "month": ts.dt.month,
                }
            )
            .groupby("CustomerId")
            .mean()
            .reset_index()
        )
        return agg


class CategoricalAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        dummies = pd.get_dummies(df[self.cols].astype(str), prefix=self.cols)
        dummies["CustomerId"] = df["CustomerId"]
        agg = dummies.groupby("CustomerId").sum().reset_index()
        return agg


def build_and_run_pipeline(input_path: str, output_path: str):
    #  Load raw data
    df = pd.read_csv(input_path, parse_dates=["TransactionStartTime"])

    # 2. Generate feature tables
    rfm_df = RFMTransformer().transform(df)
    temp_df = TemporalAggregator().transform(df)
    cat_df = CategoricalAggregator(
        ["ProductCategory", "ChannelId", "PricingStrategy"]
    ).transform(df)

    #  Standardize RFM
    scaler = StandardScaler()
    rfm_vals = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
    rfm_scaled = pd.DataFrame(rfm_vals, columns=["Recency", "Frequency", "Monetary"])
    rfm_scaled["CustomerId"] = rfm_df["CustomerId"].values

    #  Merge all features
    merged = (
        rfm_scaled.merge(temp_df, on="CustomerId", how="left")
        .merge(cat_df, on="CustomerId", how="left")
    )

    #  Save to CSV
    merged.to_csv(output_path, index=False)
    print(f"Wrote features to {output_path}")

    #  Export feature order for API
    feature_order = [c for c in merged.columns if c != "CustomerId"]
    with open("src/api/feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)
    print("Wrote feature order to src/api/feature_order.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate feature file from raw transactions and emit feature_order.json"
    )
    parser.add_argument(
        "--input", required=True, help="Path to raw CSV (data/raw/data.csv)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write processed features CSV"
    )
    args = parser.parse_args()

    build_and_run_pipeline(args.input, args.output)
