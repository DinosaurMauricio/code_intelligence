import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
