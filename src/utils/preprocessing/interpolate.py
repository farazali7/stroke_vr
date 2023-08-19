import pandas as pd
from typing import List


def interpolate_frozen_values(df: pd.DataFrame, interp_cols: List[str]) -> pd.DataFrame:
    """Interpolate frozen values.

    Replaces frozen (consecutive duplicate) values in certain columns of given dataframe with interpolated ones.

    Args:
        df: Pandas DataFrame containing some rows with frozen values
        interp_cols: List of column names in df to interpolate (may contain frozen values)

    Returns:
        Pandas DataFrame where rows with frozen values are replaced by interpolated values.
    """
    for col in interp_cols:
        orig_data = df[['timeStamp', col]]

        # Find indices where values are consecutively frozen
        consecutive_dup = orig_data.ne(orig_data.shift())
        consecutive_dup = consecutive_dup[consecutive_dup.eq(False).any(axis=1)]

        # Retain only those indices that are at consecutively duplicated values
        consecutive_dup_idxs = consecutive_dup.index.to_series().diff().fillna(2) > 1
        consecutive_dup_idxs = consecutive_dup_idxs[consecutive_dup_idxs == False].index

        # Set those to NaN
        orig_data = orig_data.apply(pd.to_numeric, errors='coerce')
        orig_data.loc[:, col].loc[consecutive_dup_idxs] = pd.NA

        # Save old index
        old_index = orig_data.index

        # Linear interpolate on timeStamp
        orig_data = orig_data.set_index('timeStamp')
        orig_data = orig_data.interpolate(method='index')
        orig_data = orig_data.reset_index()
        orig_data.index = old_index

        # Set new col with interpolated values
        df.loc[:, col] = orig_data.loc[:, col]

    return df
