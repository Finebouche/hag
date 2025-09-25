from typing import Sequence, Union

from matplotlib.colors import to_rgb, to_rgba
from scipy import signal
import numpy as np
from scipy.signal.windows import gaussian

# ----------------------------- Peaks ---------------------------------
def extract_peak_frequencies(
        input_data: np.ndarray,
        sampling_rate: float,
        threshold: float,
        smooth: bool = True,
        window_length: int = 10,
        nperseg: int = 1024,
        visualize: bool = True,
    ):
    assert threshold < 1, "Threshold should be a fraction of the maximum power"
    if visualize:
        print("Frequency limit: ", np.round(sampling_rate / 2), "(Shannon sampling theorem)")
    filtered_peak_freqs = []
    max_power = 0
    max_frequency = 0
    for i in range(input_data.shape[1]):
        # Estimate power spectral density using Welch's method
        # https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
        f, Pxx_den = signal.welch(input_data[:, i], sampling_rate, nperseg=nperseg)
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

    if input_data.shape[1] == 1:
        return filtered_peak_freqs[0]
    else:
        return np.array(filtered_peak_freqs, dtype=object)

from scipy.signal import cheby2, sosfiltfilt


# -------------------------- Band edges --------------------------------
def compute_band_edges(
    filtered_peak_freqs: Union[np.ndarray, Sequence[np.ndarray]],
    fs: float,
    min_bandwidth_hz: float = 1.0,
    eps: float = 1e-6,
):
    """Compute per-peak [low, high] from midpoints; enforce a minimum bandwidth."""
    nyq = fs / 2.0

    # Normalize to list-of-arrays
    if isinstance(filtered_peak_freqs, np.ndarray) and filtered_peak_freqs.dtype != object:
        freq_lists = [filtered_peak_freqs]
    else:
        freq_lists = list(filtered_peak_freqs)

    lows, highs = [], []
    for p in freq_lists:
        p = np.asarray(p, float)
        p = np.unique(p[(p > 0) & (p < nyq - eps)])
        if p.size == 0:
            lows.append(np.array([], float))
            highs.append(np.array([], float))
            continue

        bounds = np.r_[0.0, p, nyq]
        mids = (bounds[1:] + bounds[:-1]) / 2.0
        low = np.clip(mids[:-1], eps, nyq - eps)
        high = np.clip(mids[1:], eps, nyq - eps)

        # Enforce minimum bandwidth around the actual peak centers
        bw = high - low
        narrow = bw < min_bandwidth_hz
        if np.any(narrow):
            half = min_bandwidth_hz / 2.0
            low[narrow] = np.maximum(p[narrow] - half, eps)
            high[narrow] = np.minimum(p[narrow] + half, nyq - eps)

        valid = high > (low + eps)
        lows.append(low[valid]); highs.append(high[valid])

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

    low_list, high_list = compute_band_edges(filtered_peak_freqs, sampling_rate, min_bandwidth_hz)

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
    sampling_rate: float,
    threshold: float = 0.5,
    smooth: bool = True,
    window_length: int = 10,
    nperseg: int = 1024,
    visualize: bool = False,
    filter_order: int = 6,
    min_bandwidth_hz: float = 1.0,
):
    peaks = extract_peak_frequencies(
        input_data=np.asarray(X), sampling_rate=sampling_rate, threshold=threshold,
        smooth=smooth, window_length=window_length, nperseg=nperseg, visualize=visualize
    )
    return filter_instance_frequencies(
        X, peaks, sampling_rate, order=filter_order, min_bandwidth_hz=min_bandwidth_hz
    )


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # --- 1) Make a toy multi-channel signal -------------------------------
    fs = 1000.0         # Hz
    dur = 2.0           # seconds
    t = np.arange(int(dur * fs)) / fs
    rng = np.random.default_rng(0)

    # Channel 1 has peaks ~50, 120, 220 Hz
    x1 = (
        1.0 * np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
        + 0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * rng.standard_normal(t.shape)
    )

    # Channel 2 has peaks ~80, 150 Hz
    x2 = (
        0.8 * np.sin(2 * np.pi * 80 * t)
        + 0.6 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * rng.standard_normal(t.shape)
    )

    X = np.column_stack([x1, x2])  # (n_samples, n_channels)

    # --- 2) Detect peaks, build band edges (optional: print them) ----------
    peaks = extract_peak_frequencies(
        input_data=X,
        sampling_rate=fs,
        threshold=0.4,        # relative power threshold for peak selection
        smooth=True,
        window_length=10,
        nperseg=1024,
        visualize=True,      # set True if you want the PSD plot
    )

    # Optional: inspect the computed band edges
    low_list, high_list = compute_band_edges(peaks, fs, min_bandwidth_hz=2.0)
    print("\nDetected peaks & band edges per channel:")
    for ch, (p, lows, highs) in enumerate(zip(peaks, low_list, high_list)):
        pairs = [f"[{l:.1f}, {h:.1f}]" for l, h in zip(lows, highs)]
        print(f"  Ch{ch}: peaks ~ {np.round(p, 1)} Hz")
        print(f"       bands: {', '.join(pairs)}")

    # --- 3) Filter into n_component time series ----------------------------
    Y = filter_instance_frequencies(
        X=X,
        filtered_peak_freqs=peaks,
        sampling_rate=fs,
        order=6,
        stopband_atten_db=20.0,
        min_bandwidth_hz=2.0,
    )

    print("\nShapes:")
    print("  Input X:", X.shape)         # (n_samples, n_channels)
    print("  Output Y:", Y.shape)        # (n_samples, total_components)

    # --- 4) One-liner that does (2)+(3) if you prefer ----------------------
    Y2 = process_instance_func(
        X,
        sampling_rate=fs,
        threshold=0.4,
        smooth=True,
        window_length=10,
        nperseg=1024,
        visualize=False,
        filter_order=6,
        min_bandwidth_hz=2.0,
    )
    assert np.allclose(Y, Y2), "Sanity check: both paths should match."

    # --- 5) (Optional) Quick plots -----------------------------------------
    # Show Channel 1 and its first two extracted components (if present)
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, X[:, 0])
    plt.title("Original - Channel 1")
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude")

    if Y.shape[1] >= 1:
        plt.subplot(3, 1, 2)
        plt.plot(t, Y[:, 0])
        plt.title("Extracted Component 1")
        plt.xlabel("Time [s]"); plt.ylabel("Amplitude")

    if Y.shape[1] >= 2:
        plt.subplot(3, 1, 3)
        plt.plot(t, Y[:, 1])
        plt.title("Extracted Component 2")
        plt.xlabel("Time [s]"); plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()