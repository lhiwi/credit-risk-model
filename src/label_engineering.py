import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple


def create_proxy_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Given a customer-level feature table `df` and RFM columns in `feature_cols`,
    cluster with KMeans into `n_clusters` and assign is_high_risk=1 to the cluster
    with the largest mean Recency (feature_cols[0]) and smallest other two means.
    Returns a new DataFrame with an added 'is_high_risk' column.
    """
    # 1. Fit KMeans
    X = df[feature_cols].values
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    # 2. Compute cluster means
    summary = df.groupby("cluster")[feature_cols].mean()

    # 3. Identify high-risk cluster:
    #    highest Recency (first col), tie-break by lowest Frequency+Monetary
    rec_col, freq_col, mon_col = feature_cols
    # find cluster with max recency
    high_risk = summary[rec_col].idxmax()

    # 4. Assign proxy label
    df["is_high_risk"] = (df["cluster"] == high_risk).astype(int)
    return df.drop(columns=["cluster"])


def save_labeled_features(
    input_path: str,
    output_path: str,
    feature_cols: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> None:
    """
    Load features from CSV, apply create_proxy_label, and save to CSV.
    """
    df = pd.read_csv(input_path)
    labeled = create_proxy_label(df, feature_cols, n_clusters, random_state)
    labeled.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create proxy default labels via RFM clustering"
    )
    parser.add_argument("--input", required=True, help="Path to feature CSV")
    parser.add_argument("--output", required=True, help="Path to write labeled CSV")
    parser.add_argument(
        "--features",
        nargs=3,
        default=["Recency", "Frequency", "Monetary"],
        help="Three RFM feature column names",
    )
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    save_labeled_features(
        args.input, args.output, args.features, args.n_clusters, args.random_state
    )
