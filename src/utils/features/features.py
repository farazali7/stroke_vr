import numpy as np
import scipy
from typing import Optional


def rms(data: np.ndarray) -> np.ndarray:
    """Compute root mean square.

    Computes root mean square value on data.

    Args:
        data: Array of data

    Returns:
        RMS values for data.
    """
    return np.sqrt(np.mean(data**2, axis=1))


def magnitude(data: np.ndarray) -> np.ndarray:
    """Compute magnitude (L2 norm).

    Computes magnitude of data using L2 (Euclidean) norm.

    Args:
        data: Array of data

    Returns:
        Magnitude values for data.
    """
    magnitude_across_xyz = np.linalg.norm(data, ord=2, axis=2)
    magnitude_across_windows = np.linalg.norm(magnitude_across_xyz, ord=2, axis=1)
    return magnitude_across_windows


def std(data: np.ndarray) -> np.ndarray:
    """Compute standard deviation.

    Computes standard deviation of data.

    Args:
        data: Array of data

    Returns:
        Standard deviation values for data.
    """
    return np.std(data, axis=1)


def skew(data: np.ndarray) -> np.ndarray:
    """Compute skew.

    Computes skew of data.

    Args:
        data: Array of data

    Returns:
        Skew values for data.
    """
    return scipy.stats.skew(data, axis=1)


def kurtosis(data: np.ndarray) -> np.ndarray:
    """Compute kurtosis.

    Computes kurtosis of data.

    Args:
        data: Array of data

    Returns:
        Kurtosis values for data.
    """
    return scipy.stats.kurtosis(data, axis=1)


def dimensionless_jerk(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Compute dimensionless jerk.

    Computes dimensionless jerk of data.

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Dimensionless jerk values for data.
    """
    dt = 1./sampling_rate

    velocity = np.gradient(data, dt, axis=1)
    acceleration = np.gradient(velocity, dt, axis=1)
    jerk = np.gradient(acceleration, dt, axis=1)

    mean_velocity = np.mean(velocity, axis=1)

    movement_dur = len(data)*dt
    scale = pow(movement_dur, 3) / pow(mean_velocity, 2)

    dimless_jerk = np.sum(pow(jerk, 2), axis=1) * dt * -scale

    return dimless_jerk


def sparc(data: np.ndarray, sampling_rate: int, fc: Optional[int] = 10,
          amp_thresh: Optional[int] = 0.05) -> np.ndarray:
    """Compute SPectral ARC length.

    Computes SPARC metric of data.
    Adapted from: https://github.com/siva82kb/SPARC/blob/master/scripts/smoothness.py

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data
        fc: Max cutoff frequency for SPARC
        amp_thresh: Normalized amplitude threshold for determining cutoff frequency

    Returns:
        SPARC values for data.
    """
    dt = 1. / sampling_rate

    velocity = np.gradient(data, dt, axis=1)

    # Normalize magnitude spectrum
    mag_spectrum = abs(np.fft.fft(velocity, axis=1))
    mag_spectrum_max = np.max(mag_spectrum, axis=1)[:, np.newaxis, :]

    # Re-assign shape for proper division
    mag_spectrum_max = np.tile(mag_spectrum_max, (1, mag_spectrum.shape[1], 1))
    mag_spectrum_normalized = mag_spectrum / mag_spectrum_max

    # Remove negative frequencies
    freqs = np.fft.fftfreq(n=mag_spectrum_normalized.shape[1], d=dt)
    pos_freqs = freqs[:len(freqs)//2]

    # Select indices from spectrum that meet low-pass cutoff freq excluding DC offset
    freq_idxs = np.where(pos_freqs <= fc)[0][1:]
    mag_spectrum_normalized = mag_spectrum_normalized[:, freq_idxs, :]

    # Determine adaptive cutoff req - TODO: Implement after
    # amp_idxs = np.where(mag_spectrum_normalized >= amp_thresh)[1:]

    # Integrate (discrete sum)
    dw = freqs[1]  # Assume equivalent spaced frequencies (i.e., 0 to w1 = dw)
    mag_spectrum_grad = np.gradient(mag_spectrum_normalized, dw, axis=1)
    bias = pow(1./fc, 2)
    mag_spectrum_grad_sq = pow(mag_spectrum_grad, 2)
    sparc_vals = -np.sum(pow(bias + mag_spectrum_grad_sq, 0.5), axis=1)

    return sparc_vals

