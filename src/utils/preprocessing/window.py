import numpy as np
from numpy.lib.stride_tricks import as_strided


def window_data(data, window_size=200, overlap_size=100, remove_short=True, flatten_inside_window=False):
    """Window data based on size and overlap.

    Windowing function to split data based on window size and overlap

    Args:
        data: Array of data
        window_size: Integer, number of samples in one window
        overlap_size: Integer, number of overlapping samples between windows
        remove_short: Boolean, set True to remove (last) shorter window
        flatten_inside_window: Boolean, set True to flatten window size dimension with outer dimension

    Returns:
        Windowed array view
    """
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # Calculate the number of overlapping windows that fit into the data
    stride = window_size - overlap_size  # how much each window shifts by
    # index of last row where a valid window could start divided by the
    # stride to get how many steps can be taken until that last index is reached
    # and +1 to account for final window that may begin after that last index
    num_windows = (data.shape[0] - window_size) // stride + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, remove last shorter window
    if overhang != 0 and remove_short:
        data = data[:-overhang]

    if not data.data.contiguous:
        data = np.ascontiguousarray(data)

    window_shape = (num_windows, window_size, data.shape[1])
    window_strides = (data.strides[0]*stride, data.strides[0], data.strides[1])
    ret = as_strided(
            data,
            shape=window_shape,
            strides=window_strides,
            writeable=False
            )

    if flatten_inside_window:
        ret = ret.reshape((num_windows, -1))

    return ret
