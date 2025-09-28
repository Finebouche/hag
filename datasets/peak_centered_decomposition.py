from typing import Sequence, Union

from matplotlib.colors import to_rgb, to_rgba
from scipy import signal
import numpy as np
from scipy.signal.windows import gaussian

from matplotlib import pyplot as plt

from scipy.signal import cheby2, sosfiltfilt

# ----------------------------- Peaks ---------------------------------
def extract_peak_frequencies(
        input_data: np.ndarray,
        is_instances_classification: bool,
        sampling_rate: float,
        threshold: float = 1e-5,
        smooth: bool = True,
        window_length: int = 10,
        nperseg: int = 1024,
        visualize: bool = True,
    ):
    # if is_instances_classification use the concatenated data to find peaks
    if is_instances_classification:
        input_concat = np.concatenate(input_data, axis=0)
    else:
        input_concat = input_data

    assert threshold < 1, "Threshold should be a fraction of the maximum power"
    if visualize:
        print("Frequency limit: ", np.round(sampling_rate / 2), "(Shannon sampling theorem)")
    filtered_peak_freqs = []
    max_power = 0
    max_frequency = 0
    print(input_concat.shape)
    for i in range(input_concat.shape[1]):
        # Estimate power spectral density using Welch's method
        # https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
        f, Pxx_den = signal.welch(input_concat[:, i], sampling_rate, nperseg=nperseg)
        # Smoothing the Power Spectral Density (Pxx_den) before peak detection helps in emphasizing
        # the more significant, broader peaks that are often of greater interest in signal processing tasks.
        if smooth:
            # Create a Gaussian window
            gaus_window = gaussian(window_length, std=7)
            gaus_window /= np.sum(gaus_window)
            # Apply the Gaussian filter
            Pxx_den = signal.convolve(Pxx_den, gaus_window, mode='same')
        # Find the peaks in the power spectral density
        peak_indices, _ = signal.find_peaks(Pxx_den)
        peak_freqs = f[peak_indices]
        peak_powers = Pxx_den[peak_indices]
        # Define a power threshold and select the peaks based on that threshold
        relative_threshold = threshold * np.max(Pxx_den[peak_indices])
        filtered_peak_freqs.append(peak_freqs[peak_powers > relative_threshold])

        if visualize:
            lines = plt.semilogy(f, Pxx_den)
            line_color = to_rgb(lines[0].get_color())
            darkened_color = to_rgba([x * 0.7 for x in line_color])
            plt.plot(peak_freqs, Pxx_den[peak_indices], "o", color=darkened_color, markersize=4, label=i)

        # Calculate the maximum power peak and the maximum frequency
        if np.max(Pxx_den[peak_indices]) > max_power:
            max_power = np.max(Pxx_den[peak_indices])
        if np.max(peak_freqs[peak_powers > threshold * max_power * 1e-2]) > max_frequency:
            max_frequency = np.max(peak_freqs[peak_powers > (threshold * max_power * 1e-2)])

    if visualize:
        plt.ylim([threshold * max_power * 1e-2, max_power * 1e1])
        plt.xlim([0, max_frequency])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        # Add threshold lines
        plt.axhline(threshold * max_power, color='g', linestyle='--', label='P threshold')
        plt.axvline(sampling_rate / 2, color='r', linestyle='--', label='f limit')
        plt.legend()  # Show legend with the threshold line
        plt.show()

    if input_concat.shape[1] == 1:
        return filtered_peak_freqs[0]
    else:
        return np.array(filtered_peak_freqs, dtype=object)



# -------------------------- Band edges --------------------------------
def compute_band_edges(
    filtered_peak_freqs: Union[np.ndarray, Sequence[np.ndarray]],
    sampling_rate: float,
    X: Union[np.ndarray, Sequence[np.ndarray]],
    min_bandwidth_hz: float = 0.0,
    eps: float = 1e-6,
):
    """
    For each channel's peak list p, build bands by placing edges at the midpoints
    between neighboring peaks (i.e., half the inter-peak distance).
    Returns (lows, highs) as lists of np.ndarrays, one array per channel.
    """
    nyq = float(sampling_rate) / 2.0

    # Normalize peaks to list-of-arrays
    if isinstance(filtered_peak_freqs, np.ndarray) and filtered_peak_freqs.dtype != object:
        per_ch = [np.asarray(filtered_peak_freqs, float)]
    else:
        per_ch = [np.asarray(p, float) for p in list(filtered_peak_freqs)]

    # Determine #channels from X, then broadcast single peak-set if needed
    X_arr = np.asarray(X)
    n_channels = 1 if X_arr.ndim == 1 else int(X_arr.shape[1])
    if len(per_ch) == 1 and n_channels > 1:
        per_ch = per_ch * n_channels

    lows, highs = [], []
    for p in per_ch:
        # clean & sort peaks strictly inside (0, nyq)
        p = np.asarray(p, float)
        p = np.unique(p[(p > eps) & (p < nyq - eps)])
        if p.size == 0:
            lows.append(np.array([], float))
            highs.append(np.array([], float))
            continue

        # half-distance bands around each peak
        p.sort()
        extended = np.r_[0.0, p, nyq]          # [0, f1, f2, ..., nyq]
        halfspan = np.diff(extended) / 2.0     # length = len(p)+1
        low = p - halfspan[:-1]
        high = p + halfspan[1:]

        # enforce minimum bandwidth if requested
        if min_bandwidth_hz > 0:
            bw = high - low
            narrow = bw < min_bandwidth_hz
            if np.any(narrow):
                half = min_bandwidth_hz / 2.0
                low[narrow] = p[narrow] - half
                high[narrow] = p[narrow] + half

        # clip to (0, nyq) and ensure valid intervals
        low = np.clip(low, eps, nyq - eps)
        high = np.clip(high, eps, nyq - eps)
        valid = high > (low + eps)

        lows.append(low[valid])
        highs.append(high[valid])

    return lows, highs


# -------------------------- Filtering ---------------------------------
def band_filter(x, low_cut, high_cut, fs, order=6, stopband_atten_db=20.0):
    sos = cheby2(order, stopband_atten_db, [low_cut, high_cut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x).flatten()

def filter_instance_frequencies(
    X: Union[np.ndarray, Sequence[np.ndarray]],
    filtered_peak_freqs: Union[np.ndarray, Sequence[np.ndarray]],
    sampling_rate: float,
    order: int = 6,
    min_bandwidth_hz: float = 1.0,
    stopband_atten_db: float = 20.0,
):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, n_channels = X.shape

    low_list, high_list = compute_band_edges(filtered_peak_freqs, sampling_rate, X)

    # Single-dimension peak list given for mono? Normalize to list-of-lists
    if n_channels == 1 and (isinstance(filtered_peak_freqs, np.ndarray) and filtered_peak_freqs.dtype != object):
        low_list = [low_list[0]]
        high_list = [high_list[0]]

    # If peaks were not provided per channel, assume same peak set for all
    if len(low_list) == 1 and n_channels > 1:
        low_list, high_list = low_list * n_channels, high_list * n_channels

    # Apply filters and stack components
    per_chan_components = []
    for ch in range(n_channels):
        xch = X[:, ch]
        lows = low_list[ch]
        highs = high_list[ch]
        if lows.size == 0:
            continue  # no components in this channel

        bands = [
            band_filter(xch, float(l), float(h), sampling_rate,
                        order=order, stopband_atten_db=stopband_atten_db)
            for l, h in zip(lows, highs)
        ]
        # (n_samples, n_peaks_ch)
        per_chan_components.append(np.stack(bands, axis=1))

    if not per_chan_components:
        # No valid bands anywhere -> return zeros to keep shape predictable
        return np.zeros((n_samples, 0), dtype=X.dtype)

    return np.concatenate(per_chan_components, axis=1)


# ------------------------ Peak-centered filter-bank decomposition --------------------------
def process_instance_func(
    X: Union[np.ndarray, Sequence[np.ndarray]],
    is_instances_classification,
    sampling_rate: float,
    peaks: Union[np.ndarray, Sequence[np.ndarray]],
    filter_order: int = 6,
    min_bandwidth_hz: float = 1.0,
):
    # Filter each instance
    if is_instances_classification:
        X_filtered = []
        for instance in X:
            Y_inst = filter_instance_frequencies(
                X=instance,
                filtered_peak_freqs=peaks,
                sampling_rate=sampling_rate,
                order=filter_order,
                min_bandwidth_hz=min_bandwidth_hz,
            )
            X_filtered.append(Y_inst)
        return X_filtered
    else:
        X_filtered = filter_instance_frequencies(
            X=X,
            filtered_peak_freqs=peaks,
            sampling_rate=sampling_rate,
            order=filter_order,
            min_bandwidth_hz=min_bandwidth_hz,
        )
        return X_filtered