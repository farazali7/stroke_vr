import pandas as pd


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

    return df


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

