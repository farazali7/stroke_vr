import pandas as pd
import pickle
from typing import List

from config import cfg


def count_frozen_values(df: pd.DataFrame, cols: List[str]) -> float:
    """ Count instances of data where value is constant across successive rows.

    Counts number of rows in given column of pandas DataFrame that have exact same value as the previous row.

    Args:
        df: Pandas DataFrame containing one column of data
        cols: List of column names to determine frozen values from

    Returns:
        Number percentage of rows with repeated values
    """
    df = df[cols]

    # Find rows in any column that are not equal to a shifted (by 1) df
    consecutive_dup = df.ne(df.shift())
    consecutive_dup = consecutive_dup[consecutive_dup.eq(False).any(axis=1)]

    # Keep only rows where index is actually consecutive
    mask = consecutive_dup.index.to_series().diff() > 1
    consecutive_dup = consecutive_dup[~mask]

    num_frozen = len(consecutive_dup)

    perc_frozen = round(num_frozen / len(df), 2)

    return perc_frozen


if __name__ == '__main__':
    # Load raw data
    data_path = cfg['PATHS']['RAW']
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    # All potentially affected cols
    all_frozen_cols = ['HMD_localPosition',
                             'HMD_localRotation',
                             'LeftController_localPosition',
                             'LeftController_localRotation',
                             'RightController_localPosition',
                             'RightController_localRotation']
    all_frozen_cols = [f'{c}_{suffix}' for c in all_frozen_cols for suffix in ['x', 'y', 'z']]

    all_perc_frozen = count_frozen_values(df, all_frozen_cols)

    # HMD frozen
    hmd_frozen_cols = ['HMD_localPosition',
                             'HMD_localRotation']
    hmd_frozen_cols = [f'{c}_{suffix}' for c in hmd_frozen_cols for suffix in ['x', 'y', 'z']]

    hmd_perc_frozen = count_frozen_values(df, hmd_frozen_cols)

    # Controller frozen
    controller_frozen_cols = ['LeftController_localPosition',
                             'LeftController_localRotation',
                             'RightController_localPosition',
                             'RightController_localRotation']
    controller_frozen_cols = [f'{c}_{suffix}' for c in controller_frozen_cols for suffix in ['x', 'y', 'z']]

    controller_perc_frozen = count_frozen_values(df, controller_frozen_cols)

    # Frozen by col name
    frozen_dict = {}
    for c in all_frozen_cols:
        frozen_dict[c] = count_frozen_values(df, [c])

    print('Done')
