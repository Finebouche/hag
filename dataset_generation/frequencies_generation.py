import numpy as np
import scipy.signal as signal


# take a time serie as entry and return the optimal number of band-pass filters
def optimal_n(time_serie, sampling_frequency):
    # Compute the FFT of the time serie
    fft = np.fft.fft(time_serie)
    # Compute the frequencies of the FFT
    fft_freq = np.fft.fftfreq(len(time_serie), d=1 / sampling_frequency)
    # Compute the main frequencies of the time serie
    main_freq = fft_freq[np.argmax(np.abs(fft))]
    # Compute the optimal number of band-pass filters
    n = int(np.ceil(2 * main_freq))
    return n


# take a time serie as entry and return the optimal sampling frequency
def optimal_sampling_frequency(time_serie):
    # Compute the FFT of the time serie
    fft = np.fft.fft(time_serie)
    # Compute the frequencies of the FFT
    fft_freq = np.fft.fftfreq(len(time_serie))
    # Compute the main frequencies of the time serie
    main_freq = fft_freq[np.argmax(np.abs(fft))]
    # Compute the optimal sampling frequency
    sampling_frequency = 2 * main_freq
    return sampling_frequency


# function that takes a one dimensional time serie as entry and outputs compute n band-pass filtered around the main frequencies
# of the time serie. The main frequencies are computed using the FFT.
# The function returns a 2D array of shape (n, len(time_serie)) where n is the number of band-pass filters.
# The function also returns the main frequencies of the time serie.
# The function also returns the FFT of the time serie.
def frequencies_generation(time_serie, sampling_frequency, n):
    # Compute the FFT of the time serie
    fft = np.fft.fft(time_serie)
    # Compute the frequencies of the FFT
    fft_freq = np.fft.fftfreq(len(time_serie), d=1 / sampling_frequency)
    # Compute the main frequencies of the time serie
    main_freq = fft_freq[np.argmax(np.abs(fft))]
    # Compute the frequencies of the band-pass filters
    freqs = np.linspace(main_freq - 0.1, main_freq + 0.1, n)
    # Compute the band-pass filters
    filters = [signal.butter(10, [freqs[i] - 0.01, freqs[i] + 0.01], 'bandpass', fs=sampling_frequency, output='sos')
               for i in range(n)]
    # Apply the band-pass filters to the time serie
    filtered_time_series = [signal.sosfiltfilt(filters[i], time_serie) for i in range(n)]
    # Return the filtered time series, the main frequencies and the FFT of the time serie
    return filtered_time_series, main_freq, fft
