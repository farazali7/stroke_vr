import pandas as pd
import os
import pickle
from typing import List, Optional
import numpy as np
from tqdm import tqdm
import logging

from config import cfg


def load_subject_data(base_path: str, subject_id: int, sessions: List[int]) -> pd.DataFrame:
    """Retrieve subject data.

    Loads and returns data for specified subject into a DataFrame.

    Args:
        base_path: Path to base data directory
        subject_id: Integer specifying subject number
        sessions: List of session numbers for which to retrieve data from

    Returns:
        A Pandas DataFrame containing subject's data
    """
    subject_dir = 'S ' + str(subject_id)
    file_paths = [os.path.join(base_path, subject_dir, f'SESION {x}.csv') for x in sessions]
    all_sessions_dfs = []
    for i, file_path in enumerate(file_paths):
        session_df = pd.DataFrame()
        if not os.path.exists(file_path):
            logging.warning(f'File path: {file_path} does not exist... '
                            f'Returning empty DataFrame')
        else:
            session_df = pd.read_csv(file_path)

            # Apply fix for semicolon delimited files
            if len(session_df.columns) == 1:
                # Skip rows that don't have proper length column data
                session_df = pd.read_csv(file_path, sep=';', on_bad_lines='warn')
                session_df.dropna(inplace=True)

        # Set all non subject, session, time cols to numeric dtypes
        cols = session_df.columns[3:]
        session_df[cols] = session_df[cols].apply(pd.to_numeric, errors='coerce')
        # session_df = session_df.apply(pd.to_numeric, errors='coerce')
        session_df.insert(loc=0, column='Subject', value=subject_id)
        session_df.insert(loc=1, column='Session', value=i + 1)

        all_sessions_dfs.append(session_df)

    subject_df = pd.concat(all_sessions_dfs, ignore_index=True)

    return subject_df


def create_raw_df(base_path: str, subject_ids: List[int], sessions: List[int],
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """Generate DataFrame containing all raw data.

    Loads and creates a Pandas DataFrame containing raw data from all specified subjects and sessions.

    Args:
        base_path: Path to base data directory
        subject_ids: List of subject ids for which to retrieve
        sessions: List of session numbers for which to retrieve data from
        save_path: Optional string specifying path to write final DataFrame to disk

    Returns:
        A Pandas DataFrame containing all subjects' data
    """
    all_subject_dfs = []
    for subject_id in tqdm(subject_ids, total=len(subject_ids)):
        subject_df = load_subject_data(base_path, subject_id, sessions)

        # Interpolate missing data for subject 6
        if subject_id == 6:
            interpolated = subject_df[subject_df['Session'] == 11]
            interpolated['Session'] = 12
            interpolated.iloc[:, 5:] = np.nan  # Skip over subject, session, time cols
            subject_df = pd.concat([subject_df, interpolated], ignore_index=True)
            subject_df = subject_df.set_index('timeStamp')  # Will interpolate based on these values
            subject_df = subject_df.interpolate(method='index')
            subject_df = subject_df.reset_index()
            logging.info(f'Interpolated values for subject: {subject_id}')

        all_subject_dfs.append(subject_df)

    raw_df = pd.concat(all_subject_dfs, ignore_index=True)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(raw_df, f)

    return raw_df


if __name__ == "__main__":
    raw_df = create_raw_df(base_path=cfg['PATHS']['DATA_DIR'],
                           subject_ids=cfg['SUBJECTS'],
                           sessions=cfg['SESSIONS'],
                           save_path=cfg['PATHS']['RAW'])

    logging.info('Done')
