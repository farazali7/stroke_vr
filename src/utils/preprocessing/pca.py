import pandas as pd
from sklearn.decomposition import PCA


def add_pca_cols(data: pd.DataFrame, n_components=3) -> pd.DataFrame:
    """Adds PCA columns to data.

    Args:
        data: DataFrame of all subjects' and sessions' data.
        n_components: Number of principal components to retain (could also be explained variance fraction in [0, 1])

    Returns:
        DataFrame containing original data & PCA vector columns.
    """
    cols_to_drop = ['Subject', 'Session', 'Window', 'perceived_enjoyment', 'perceived_exertion', 'game_performance']
    features_data = data.drop(columns=cols_to_drop).values
    pca = PCA(n_components=n_components)
    pca_vectors = pca.fit_transform(features_data)
    pca_df = pd.DataFrame(pca_vectors)
    pca_df.columns = [f'PC{i}' for i in range(1, pca_vectors.shape[1] + 1)]

    data = pd.concat([data, pca_df.set_index(data.index)], axis=1)

    return data
