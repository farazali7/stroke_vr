import logging
import os
import pickle
import json
from typing import List, Any
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial

import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.utils.preprocessing import window_data, apply_filter, add_response_variable
from src.utils.features import *

from config import cfg


def extract_feature_set(data: np.ndarray, feature_list: List[str], sampling_rate: int) -> pd.DataFrame:
    """Extract a set of features from data.

    Windows the raw data and computes the feature data for each window before returning a list of
    feature data for all windows.

    Args:
        data: Array containing raw data
        feature_list: List of strings specifying feature names
        sampling_rate: Sampling rate of data

    Returns:
        DataFrame of feature data
    """
    features = pd.DataFrame()
    for feature_name in feature_list:
        feature_fn = eval(feature_name)

        if feature_name in ['dimensionless_jerk', 'sparc']:
            feature_data = feature_fn(data, sampling_rate)
        else:
            feature_data = feature_fn(data)

        if feature_data.ndim == 2:  # Average across xyz axes if necessary
            feature_data = np.mean(feature_data, axis=1)

        features[feature_name] = feature_data

    return features


def preprocess_session_data(session_df: pd.DataFrame, cols: dict, windowing_args: dict,
                            filtering_args: dict, feature_extraction_args: dict) -> pd.DataFrame:
    """Preprocesses data within one game session.

    Performs filtering, windowing, feature extraction, and normalization from given session data
    based on provided relevant columns of raw data.

    Args:
        session_df: DataFrame consisting of session data
        cols: Dictionary of data type (keys) and list of column names (values) from which to compute features
        windowing_args: Dictionary of kwargs for window_data function (size and overlap specified in seconds)
        filtering_args: Dictionary of kwargs for filtering (function name and args)
        feature_extraction_args: Dictionary of kwargs for extract_feature_set function

    Returns:
        DataFrame of features from session
    """
    session_features = pd.DataFrame()  # TODO MAKE INTO DF
    # Process data from inverse model and raw data columns
    for col in cols['INVERSE_AND_RAW']:
        dimension_cols = [col+'_'+dim for dim in ['x', 'y', 'z']]
        raw_data = session_df[dimension_cols]
        raw_data = raw_data.to_numpy()

        # Filter data
        filtered_data = apply_filter(raw_data, filtering_args)

        # Window data
        # Determine window size and overlap in samples
        window_size = windowing_args['WINDOW_SIZE']
        window_overlap = windowing_args['WINDOW_OVERLAP']
        sampling_rate = windowing_args['sampling_rate']
        window_sample_size = round(window_size * sampling_rate)
        window_sample_overlap = round(window_overlap * sampling_rate)
        windowed_data = window_data(filtered_data, window_size=window_sample_size,
                                    overlap_size=window_sample_overlap)

        # Extract features
        feature_data = extract_feature_set(windowed_data, **feature_extraction_args)

        # Rename columns to match data column name
        new_names = [col+'-'+x for x in feature_data.columns]
        feature_data.columns = new_names

        session_features = pd.concat([session_features, feature_data], axis=1)

    # # Process data from game data columns
    # for col in cols['GAME']:
    #     raw_data = session_df[col]
    #     raw_data = raw_data.to_numpy()
    #     feature_data = extract_feature_set(raw_data, **feature_extraction_args)
    #     session_features.append(feature_data)

    # Drop na
    session_features = session_features.replace([np.inf, -np.inf], np.nan)
    session_features = session_features.dropna().reset_index()
    session_features.drop('index', axis=1, inplace=True)

    # Normalize session data
    cols_temp = session_features.columns
    x = session_features.values
    ss = preprocessing.MinMaxScaler()
    x_scaled = ss.fit_transform(x)
    session_features = pd.DataFrame(x_scaled, columns=cols_temp)

    # Set subject, session, and window numbers
    session_features.insert(0, 'Window', session_features.index)
    session_features.insert(0, 'Session', session_df.Session.iloc[0])
    session_features.insert(0, 'Subject', session_df.Subject.iloc[0])

    return session_features


def preprocess_data(data_path: str, cols: dict, windowing_args: dict, filtering_args: dict,
                    feature_extraction_args: dict,  response_var_path: str,
                    response_var_set: List[str], save_dir: Optional[str] = None) -> pd.DataFrame:
    """Preprocess specified DataFrame by extracting features and appending response variables.

    Extract features for given DataFrame by subject and session while also appending desired response
    variables.

    Args:
        data_path: Path to raw DataFrame file (ex. stored in .pkl file)
        cols: Dictionary of data type (keys) and list of column names (values) from which to compute features
        windowing_args: Dictionary of kwargs for window_data function (size and overlap specified in seconds)
        filtering_args: Dictionary of kwargs for filtering (function name and args)
        feature_extraction_args: Dictionary of kwargs for extract_feature_set function
        response_var_path: Path to file containing response variable data
        response_var_set: List of response variables to include
        save_dir: Optional string specifying directory to save final feature DataFrame version to disk

    Returns:
        DataFrame containing processed data with features and response variables columns appended
    """
    df = None
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    # Go through each subject
    logging.info('Extracting feature data by subject...')
    subjects_gb = df.groupby('Subject')
    subject_dfs = [subjects_gb.get_group(x) for x in subjects_gb.groups]
    features = []
    for subject_df in tqdm(subject_dfs):
        # Process each session
        sessions_gb = subject_df.groupby('Session')
        session_dfs = [sessions_gb.get_group(x) for x in sessions_gb.groups]

        # TODO: Remove just for debugging
        # for i in range(10):
        #     res = preprocess_session_data(session_dfs[i], cols, windowing_args, filtering_args, feature_extraction_args)

        with Pool(processes=os.cpu_count()) as pool:
            res = list(tqdm(pool.imap(partial(preprocess_session_data,
                                              cols=cols,
                                              windowing_args=windowing_args,
                                              filtering_args=filtering_args,
                                              feature_extraction_args=feature_extraction_args),
                                      session_dfs),
                            total=len(session_dfs)))
        features.append(pd.concat(res))

    features = pd.concat(features)

    # Add response variables
    for resp_var in response_var_set:
        features = add_response_variable(features, response_var_path, resp_var)

    if save_dir is not None:
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Get latest version
        versions = [int(x.replace('v', '')) for x in os.listdir(save_dir) if 'v' in x]
        if len(versions) == 0:
            versions = [0]
        new_version = max(versions)+1

        # Create latest version folder
        save_path = os.path.join(save_dir, f'v{new_version}')
        os.mkdir(save_path)

        # Save data file
        file_save_path = os.path.join(save_path, 'feature_data.pkl')
        with open(file_save_path, 'wb') as f:
            pickle.dump(features, f)

        # Also save json of config
        file_cfg = {'cols': cols,
                    'windowing_args': windowing_args,
                    'filtering_args': filter_args,
                    'feature_extraction_args': feature_extractions_args,
                    'response_variables': response_var_set}
        cfg_save_path = os.path.join(save_path, 'config.txt')
        with open(cfg_save_path, 'w') as f:
            json.dump(file_cfg, f)

    return features


if __name__ == '__main__':
    raw_df_path = cfg['PATHS']['RAW']
    cols = cfg['COLUMNS']

    windowing_args = cfg['WINDOWING']
    windowing_args['sampling_rate'] = cfg['SAMPLING_RATE']

    filter_fn = cfg['FILTER_TYPE']
    filter_args = cfg['FILTERS'][filter_fn]
    filter_args['sampling_rate'] = cfg['SAMPLING_RATE']
    filtering_args = {'filter_fn': filter_fn,
                      'args': filter_args}

    feature_extractions_args = {'feature_list': cfg['FEATURES'],
                                'sampling_rate': cfg['SAMPLING_RATE']}

    response_var_path = cfg['PATHS']['RESPONSE_VAR']
    response_var_set = cfg['RESPONSE_VARS']

    save_dir = cfg['PATHS']['FEATURE_DATA_DIR']

    processed_df = preprocess_data(raw_df_path, cols, windowing_args, filtering_args,
                                   feature_extractions_args, response_var_path,
                                   response_var_set, save_dir)

    logging.info('Done')
