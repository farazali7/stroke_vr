cfg = {
    'PATHS': {
        'DATA_DIR': 'data/',  # Directory where raw data files are held
        'RESPONSE_VAR': 'data/TABULATION.xlsx',  # File containing response variable data
        'RAW': 'data/raw_data.pkl',  # File of concatenated DataFrame of raw data
        'FEATURE_DATA_DIR': 'data/features/',  # Directory to save feature data file versions
        'FEATURE_DATA': 'data/features/v1'  # Which version of features data file to use for modelling
    },
    'SUBJECTS': [x+1 for x in range(10)],
    'SESSIONS': [x+1 for x in range(12)],
    # Columns from which to compute features, 'INVERSE_AND_RAW' have xyz components, 'GAME' data does not
    'COLUMNS': {'INVERSE_AND_RAW': ['leftArm_upperArmRotation',
                                    'leftArm_upperArmPos',
                                    'leftArm_lowerArmRotation',
                                    'leftArm_lowerArmPos',
                                    'rightArm_upperArmRotation',
                                    'rightArm_upperArmPos',
                                    'rightArm_lowerArmRotation',
                                    'rightArm_lowerArmPos',
                                    'HMD_localPosition',
                                    'HMD_localRotation',
                                    'LeftController_localPosition',
                                    'LeftController_localRotation',
                                    'RightController_localPosition',
                                    'RightController_localRotation'],
                'GAME': []},
    'FEATURES': ['rms', 'magnitude', 'std', 'skew', 'kurtosis', 'dimensionless_jerk', 'sparc'],
    'RESPONSE_VARS': ['perceived_exertion', 'perceived_enjoyment'],  # Response variables to use as targets
    'SAMPLING_RATE': 30,  # Hz
    'WINDOWING': {'WINDOW_SIZE': 1,  # Size of windows in seconds
                  'WINDOW_OVERLAP': 0.5},  # Amount of overlap between to windows in seconds},
    'FILTER_TYPE': 'butter_low_pass',
    'FILTERS': {'butter_low_pass': {'order': 10,
                                    'cutoff': 1,
                                    'axis': 0}}
}
