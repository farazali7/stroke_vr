from typing import Tuple, List

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import cfg


def perceived_exertion(df: pd.DataFrame, response_var_path: str) -> pd.DataFrame:
    """Add perceived exertion response variable data.

    Loads and processes perceived exertion response variable data.

    Args:
        df: DataFrame of data
        response_var_path: Path to file containing response variable data

    Returns:
        DataFrame containing original data and response variables adequately added in.
    """
    # Format response df
    resp_df = pd.read_excel(response_var_path, sheet_name=0)
    resp_df = resp_df.dropna(axis=1)
    resp_df = resp_df.transpose()
    resp_df.columns = ['Session', 'Game'] + [f'Subject{n}' for n in range(1, 11)]

    # Initialize response cols in data df
    df['perceived_exertion_M'] = ''
    df['perceived_exertion_T'] = ''
    df['perceived_exertion_C'] = ''

    for subject in df.Subject.unique():
        for session in df.Session.unique():
            for game in ['M', 'T', 'C']:
                # Find value
                val = resp_df.loc[(resp_df['Session'] == session) & (resp_df['Game'] == game),
                            f'Subject{subject}'].values[0]

                # Assign
                df.loc[(df['Subject']==subject) & (df['Session']==session), f'perceived_exertion_{game}'] = val

    df['perceived_exertion'] = np.where(df[['perceived_exertion_M',
                                            'perceived_exertion_T',
                                            'perceived_exertion_C']]
                                        .mean(1).astype(int) > 3, 1, 0)

    df.drop(columns=['perceived_exertion_M', 'perceived_exertion_T', 'perceived_exertion_C'], inplace=True)

    return df


def perceived_enjoyment(df: pd.DataFrame, response_var_path: str) -> pd.DataFrame:
    """Add perceived enjoyment response variable data.

    Loads and processes perceived enjoyment response variable data.

    Args:
        df: DataFrame of data
        response_var_path: Path to file containing response variable data

    Returns:
        DataFrame containing original data and response variables adequately added in.
    """
    # Format response df
    resp_df = pd.read_excel(response_var_path, sheet_name=1, header=None)
    resp_df = resp_df.drop(0, axis=1)
    resp_df = resp_df.transpose()
    resp_df = resp_df.drop(1, axis=0)

    resp_df.columns = ['Session', 'Question'] + [f'Subject{n}' for n in range(1, 11)]

    # Interpolate missing vals (Subject 9 Session 7)
    resp_df = resp_df.apply(pd.to_numeric)
    resp_df = resp_df.interpolate(method='nearest')

    # Initialize response cols in data df
    df['perceived_enjoyment_1'] = ''
    df['perceived_enjoyment_2'] = ''
    df['perceived_enjoyment_3'] = ''

    for subject in df.Subject.unique():
        for session in df.Session.unique():
            for question in [1, 2, 3]:
                # Find value
                val = resp_df.loc[(resp_df['Session'] == session) & (resp_df['Question'] == question),
                            f'Subject{subject}'].values[0]

                # Assign
                df.loc[(df['Subject']==subject) & (df['Session']==session), f'perceived_enjoyment_{question}'] = val

    df['perceived_enjoyment'] = np.where(df[['perceived_enjoyment_1',
                                             'perceived_enjoyment_2',
                                             'perceived_enjoyment_3']]
                                         .mean(1).astype(int) > 4, 1, 0)

    df.drop(columns=['perceived_enjoyment_1', 'perceived_enjoyment_2', 'perceived_enjoyment_3'], inplace=True)

    return df


def game_performance(df: pd.DataFrame, response_var_path: str) -> pd.DataFrame:
    """Add game performance response variable data.

    Loads and processes game performance response variable data.

    Args:
        df: DataFrame of data
        response_var_path: Path to file containing response variable data

    Returns:
        DataFrame containing original data and response variables adequately added in.
    """
    # Format response df
    resp_df = pd.read_excel(response_var_path, sheet_name='gm', header=None)
    resp_df = resp_df.drop(0, axis=1)
    nh = resp_df.iloc[0].tolist()
    resp_df = resp_df[1:]
    resp_df.columns = nh

    # Initialize response cols in data df
    df['game_performance'] = -1

    for subject in df.Subject.unique():
        for session in df.Session.unique():
            # Find value
            val = resp_df.loc[(resp_df['Session'] == session) & (resp_df['Subject'] == subject),
                        f'game_performance'].values[0]

            # Assign
            df.loc[(df['Subject']==subject) & (df['Session']==session), f'game_performance'] = val

    return df


def count_box_hits(data: pd.DataFrame) -> Tuple[int, int]:
    """Counts boxes hit during gameplay of a session.
    
    Count the number of boxes hit (success) and missed (fail) during a certain session of gameplay.
    
    Args:
        data: DataFrame containing data for one session of gameplay.

    Returns:
        Tuple of fails and success counts for hitting boxes.
    """
    hammer_arr = data['LastDamageForce_hammer']
    time_arr = data['timeStamp']
    _timestamp_arr = []
    previous_value = 0

    for idx, val in hammer_arr.items():
        if hammer_arr[idx] != previous_value:
            previous_value = hammer_arr[idx]
            if previous_value > 0:
                _timestamp_arr.append(time_arr[idx])

    vel_arr = []

    for idx in range(len(_timestamp_arr)-1):
        vel_arr.append(_timestamp_arr[idx + 1] - _timestamp_arr[idx])

    successes = 0
    fails = 0
    for val in vel_arr:
        if val > 5:
            successes += 1
        elif val > 1:
            fails += 1

    return fails, successes


def count_good_discs_shot(data: pd.DataFrame) -> int:
    """Count number of good movements while shooting discs in gameplay.

    Args:
        data: DataFrame of session data.

    Returns:
        Number of good movement discs shots made
    """
    moves = 0
    disc_arr = data['score_tejo']
    previous_value = 0

    for idx, val in disc_arr.items():
        if disc_arr[idx] != previous_value:
            previous_value = disc_arr[idx]
            if previous_value > 0:
                moves += 1

    return moves


def count_all_discs_shot(data: pd.DataFrame) -> int:
    """Count number of all movements while shooting discs in gameplay.

    Args:
        data: DataFrame of session data.

    Returns:
        Number of good movement discs shots made
    """
    moves = 0
    disc_arr = data['force_shot']
    previous_value = 0

    for idx, val in disc_arr.items():
        if disc_arr[idx] != previous_value:
            previous_value = disc_arr[idx]
            if previous_value > 0:
                moves += 1

    return moves


def get_disc_shot_fails(data: pd.DataFrame) -> int:
    """Get number of failed disc shots.

    Args:
        data: DataFrame of session data.

    Returns:
        Number of failed disc shots.
    """
    return count_all_discs_shot(data) - count_good_discs_shot(data)


def count_trees_cut(data: pd.DataFrame) -> Tuple[int, int, int]:
    """Count number of trees cut from right and left hands and total spawned amount.

    Args:
        data: DataFrame of session data.

    Returns:
        Tuple of trees cut by right hand & by left hand, and how many spawned in total.
    """
    right_arr = data['score_right_riding']
    right = int(right_arr.max())

    left_arr = data['score_left_riding']
    left = int(left_arr.max())

    spawn_arr = data['totalSpawn_riding']
    total_spawned = int(spawn_arr.max())

    return right, left, total_spawned


def get_trees_cut_performance(data: pd.DataFrame) -> Tuple[int, int]:
    """Get number of successes and failures for tree cutting mini-game.

    Args:
        data: DataFrame of session data.

    Returns:
        Tuple of number of successes and failures for tree cutting mini-game
    """
    right, left, _ = count_trees_cut(data)

    successes = right + left
    failures = 140 - successes

    return failures, successes


def compute_game_performance(session_dfs: List[pd.DataFrame]) -> List[int]:
    """Compute overall game performance per session.

    Computes the successes in boxes hit, discs shot, and trees cut in all sessions then computes
    and overall game performance metric. This is computed by calculating the success percentage
    per mini-game relative to the max performance of that mini-game across all sessions. The mean of success
    percentage in all three mini-games is then presented as an overall game performance indicator.

    Args:
        session_dfs: List of DataFrames where each contains data for one session of gameplay.

    Returns:
        List of game performance values for each session.
    """
    session_metrics = []
    for session_df in session_dfs:
        _, boxes_hit = count_box_hits(session_df)
        discs_shot = count_good_discs_shot(session_df)
        _, trees_cut = get_trees_cut_performance(session_df)
        session_metrics.append((boxes_hit, discs_shot, trees_cut))

    session_metrics = np.stack(session_metrics)
    # Normalize by max (turn to fraction)
    session_metrics = session_metrics / np.max(session_metrics, axis=0)

    # Get average across three mini-games
    game_performance = np.mean(session_metrics, axis=1)

    # Threshold to binary labels
    game_performance = np.where(game_performance > 0.7, 1, 0)

    return game_performance


def add_response_variable(df: pd.DataFrame, response_var_path: str,
                          response_var_name: str) -> pd.DataFrame:
    """Add a response variable based on name.

    Processes and adds respective response variable to given DataFrame.

    Args:
        df: DataFrame of data
        response_var_path: Path to file containing response variable data
        response_var_name: Name of response variable

    Returns:
        DataFrame containing original data and response variables adequately added in.
    """
    resp_var_fn = eval(response_var_name)
    new_df = resp_var_fn(df, response_var_path)

    return new_df


if __name__ == '__main__':
    # Add game metrics as additional response vars
    game_metrics_df = pd.DataFrame(columns=['Subject', 'Session', 'game_performance'])
    raw_df = pd.read_pickle('data/raw_data.pkl')
    subjects_gb = raw_df.groupby('Subject')
    subject_dfs = [subjects_gb.get_group(x) for x in subjects_gb.groups]
    for i in tqdm(range(len(subject_dfs)), total=len(subject_dfs)):
        subject_df = subject_dfs[i]
        sessions_gb = subject_df.groupby('Session')
        session_dfs = [sessions_gb.get_group(x) for x in sessions_gb.groups]
        gp = compute_game_performance(session_dfs)
        for j in range(len(gp)):
            game_metrics_df.loc[len(game_metrics_df)] = [int(i+1), int(j+1), gp[j]]

    with pd.ExcelWriter(cfg['PATHS']['RESPONSE_VAR'], engine='openpyxl', mode='a') as writer:
        game_metrics_df.to_excel(writer, sheet_name='gm')

    print('Done')
