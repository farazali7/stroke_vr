from typing import Tuple, List

import numpy as np
import pandas as pd


def slice_features_and_labels(data: pd.DataFrame, label_cols: List[str]) \
        -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Retrieve X and Y values from given dataset.

    Extracts the X values (both kinematic and PCA) and Y values (labels) from given DataFrame, as well as
    the groups (subjects).

    Args:
        data: DataFrame of features and labels
        label_cols: List of label column names

    Returns:
        Tuple of kinematic features, PCA components, DataFrame of labels with respective label cols, and groups.
    """
    groups = data['Subject'].values

    cols_to_exclude = ['Subject', 'Session', 'Window']
    data = data.drop(columns=cols_to_exclude)

    # PCA cols
    pca_cols = [c for c in data.columns if c.startswith('PC')]
    X_pca = data.loc[:, pca_cols].values

    # Separate labels
    labels = data.loc[:, label_cols]

    # Kinematic data
    X_kin = data.drop(columns=label_cols+pca_cols).values

    return X_kin, X_pca, labels, groups


