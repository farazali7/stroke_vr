import numpy as np
import scipy
from typing import Optional, Union, List


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


def _gradient(data: np.ndarray,
              dt: Union[int, float, List[int], List[float], None] = None,
              axis: int = 1) -> np.ndarray:
    """Take gradient of input data.

    Computes gradient of data at dt if dt is provided, else assumes uniform spacing between points
    in data.

    Args:
        data: Array of data
        dt: Single value for distance between measurements in data or array for points to take gradient on
        axis: Dimension to compute gradient along

    Returns:
        Gradient values for data.
    """
    return np.gradient(data, dt, axis=axis)


def _velocity(data: np.ndarray,
             dt: Union[int, float, List[int], List[float], None] = None,
             axis: int = 1) -> np.ndarray:
    """Velocity of positional data.

    Computes velocity of data.

    Args:
        data: Array of positional data
        dt: Single value for distance between measurements in data or array for points to take gradient on
        axis: Dimension to compute velocity along

    Returns:
        Velocity values for data.
    """
    return _gradient(data, dt, axis)


def mean_velocity(data: np.ndarray,
             sampling_rate: int,
             axis: int = 1) -> np.ndarray:
    """Mean velocity of positional data.

    Computes mean velocity of data.

    Args:
        data: Array of positional data
        sampling_rate: Sampling rate of data
        axis: Dimension to compute velocity along

    Returns:
        Mean velocity for data.
    """
    dt = 1. / sampling_rate
    return np.mean(_velocity(data, dt, axis), axis=1)


def _acceleration(data: np.ndarray,
                 dt: Union[int, float, List[int], List[float], None] = None,
                 axis: int = 1) -> np.ndarray:
    """Acceleration of positional data.

    Computes acceleration of data.

    Args:
        data: Array of positional data
        dt: Single value for distance between measurements in data or array for points to take gradient on
        axis: Dimension to compute acceleration along

    Returns:
        Acceleration values for data.
    """
    return _gradient(_velocity(data, dt, axis), dt, axis)


def mean_acceleration(data: np.ndarray,
             sampling_rate: int,
             axis: int = 1) -> np.ndarray:
    """Mean acceleration of positional data.

    Computes mean acceleration of data.

    Args:
        data: Array of positional data
        sampling_rate: Sampling rate of data
        axis: Dimension to compute acceleration along

    Returns:
        Mean acceleration for data.
    """
    dt = 1. / sampling_rate
    return np.mean(_acceleration(data, dt, axis), axis=1)


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

    velocity = _velocity(data, dt, axis=1)
    acceleration_vals = np.gradient(velocity, dt, axis=1)
    jerk = np.gradient(acceleration_vals, dt, axis=1)

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

    velocity = _velocity(data, dt, axis=1)

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


def spectral_entropy(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Spectral entropy feature

    Computes spectral entropy from acceleration of data (entropy in frequency domain).

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Spectral entropy values for data.
    """
    dt = 1. / sampling_rate

    acceleration = _acceleration(data, dt, axis=1)

    # Real-valued fft
    rfft = np.fft.rfft(acceleration, axis=1)
    bins = np.fft.rfftfreq(n=data.shape[1], d=dt)

    # Normalized PSD
    mag_spectrum = abs(rfft)
    psd = 2 * (mag_spectrum ** 2) / (len(bins) * sampling_rate)
    norm_psd = psd / np.sum(psd)

    # Entropy
    spectral_entropy = -1 * np.sum(norm_psd * np.log(norm_psd), axis=1)

    return spectral_entropy


def energy_acceleration(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Energy of acceleration

    Computes energy of acceleration (by Parseval's theorem this is equivalent to same computation in
    frequency domain).

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Energy of acceleration values for data.
    """
    dt = 1. / sampling_rate

    acceleration = _acceleration(data, dt, axis=1)

    acceleration_energy = np.sum(np.abs(acceleration)**2, axis=1)

    return acceleration_energy


def power_acceleration(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Mean power of acceleration.

    Computes mean power of acceleration.

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Mean power of acceleration values for data.
    """
    dt = 1. / sampling_rate

    acceleration = _acceleration(data, dt, axis=1)

    # Real-valued fft
    rfft = np.fft.rfft(acceleration, axis=1)
    bins = np.fft.rfftfreq(n=data.shape[1], d=dt)

    # PSD
    mag_spectrum = abs(rfft)
    psd = 2 * (mag_spectrum ** 2) / (len(bins) * sampling_rate)

    mean_psd = np.mean(psd, axis=1)

    return mean_psd


def spectral_centroid(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Spectral centroid on acceleration data.

    Computes spectral centroid of acceleration. This is just the weighted mean frequency.

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Spectral centroid based off acceleration values for data.
    """
    dt = 1. / sampling_rate

    acceleration = _acceleration(data, dt, axis=1)

    # Real-valued fft
    rfft = np.fft.rfft(acceleration, axis=1)
    bins = np.fft.rfftfreq(n=data.shape[1], d=dt)

    # Weighted mean frequency
    mag_spectrum = abs(rfft)
    # Ensure bins shape matches magnitude spectrum before weighting
    bins_extended = np.tile(bins[np.newaxis, ..., np.newaxis],
                            [mag_spectrum.shape[0], 1, mag_spectrum.shape[2]])
    spectral_centroid = np.average(bins_extended, axis=1, weights=mag_spectrum)

    return spectral_centroid


def spectral_bandwidth(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Spectral bandwidth on acceleration data.

    Computes spectral bandwidth of acceleration. Using p = 2 sets the computation to be a weighted
    standard deviation.

    Args:
        data: Array of data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Spectral bandwidth based off acceleration values for data.
    """
    dt = 1. / sampling_rate

    acceleration = _acceleration(data, dt, axis=1)

    # Real-valued fft
    rfft = np.fft.rfft(acceleration, axis=1)
    bins = np.fft.rfftfreq(n=data.shape[1], d=dt)

    # Weighted mean frequency
    mag_spectrum = abs(rfft)
    bins_extended = np.tile(bins[np.newaxis, ..., np.newaxis],
                            [mag_spectrum.shape[0], 1, mag_spectrum.shape[2]])
    spectral_centroid = np.average(bins_extended, axis=1, weights=mag_spectrum)

    # Weighted squared differences
    bins_extended = np.tile(bins[np.newaxis, ..., np.newaxis],
                            [spectral_centroid.shape[0], 1, spectral_centroid.shape[1]])
    centroids_extended = np.tile(spectral_centroid[:, np.newaxis, :], (1, bins.shape[0], 1))
    squared_diff = (bins_extended - centroids_extended) ** 2
    spectral_var = np.sum(mag_spectrum * squared_diff, axis=1)

    # Bandwidth (std in this case)
    spectral_bandwidth = spectral_var ** 0.5

    return spectral_bandwidth


def cross_spectral_density(data1: np.ndarray, data2: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Cross-spectral density on acceleration data.

    Computes cross-spectral density of acceleration on data from two different sensors. This is
    the cross-correlation of the psd between the two signals.


    Args:
        data1: Array of sensor 1 data (position/rotation)
        data2: Array of sensor 2 data (position/rotation)
        sampling_rate: Sampling rate of data

    Returns:
        Cross-spectral density based off acceleration values for data.
    """
    dt = 1. / sampling_rate

    acceleration1 = _acceleration(data1, dt, axis=1)
    acceleration2 = _acceleration(data2, dt, axis=1)

    # Real-valued fft
    rfft1 = np.fft.rfft(acceleration1, axis=1)
    bins1 = np.fft.rfftfreq(n=rfft1.shape[1], d=dt)

    rfft2 = np.fft.rfft(acceleration2, axis=1)
    bins2 = np.fft.rfftfreq(n=rfft2.shape[1], d=dt)

    # CSD
    csd = 2 * np.multiply(np.conjugate(rfft1), np.conjugate(rfft2)) / (len(bins1) * sampling_rate)

    mean_csd = np.mean(csd, axis=1).real

    return mean_csd
