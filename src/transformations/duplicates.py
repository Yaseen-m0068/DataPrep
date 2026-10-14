
import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    #Return a copy of the dataset with exact duplicate rows removed.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot remove duplicates from an empty dataset.")

    return df.drop_duplicates(keep="first").copy()
