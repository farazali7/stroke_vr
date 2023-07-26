from scipy.signal import butter, sosfiltfilt
import numpy as np


def butter_low_pass(data, order=4, cutoff=10, sampling_rate=100, axis=0):
    """Butterworth low-pass filter.

    Applies a Butterworth low-pass filter of the order specified and at the desired cut-off
    frequency.

    Args:
        data: Array of input data
        order: Order of filter
        cutoff: Frequency cut-off in Hz
        sampling_rate: Sampling rate of data
        axis: Axis of data to apply filter along

    Returns:
        Butterworth low-pass filtered data
    """
    sos = butter(order, cutoff, 'lowpass', fs=sampling_rate, output='sos')
    filtered = sosfiltfilt(sos, data, axis=axis)

    return filtered


def apply_filter(data: np.ndarray, filtering_args: dict) -> np.ndarray:
    """Apply a filter.

    Applies a filter based on specified function name and arguments provided

    Args:
        data: Array of data
        filtering_args: Dictionary containing function name and pertinent args.

    Returns:
        Filtered data
    """
    filter_fn = eval(filtering_args['filter_fn'])
    filter_args = filtering_args['args']
    filtered_data = filter_fn(data, **filter_args)

    return filtered_data
