import numpy as np
def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024)->float:
    """
    Identifies the primary sinusoidal component in a signal x[n]
    over time by calculting successive N-point DFTs of x[n], and
    selecting the frequency component with the highest magnitude. 

    Parameters:
    x_n - signal samples x[n] to be analyzed
    fs - sampling frequency
    N - DFT window size in number of samples 
        Defaults to 1024 samples

    Returns:
    timestamps - ndarray of floats
        Points in time at which frequency contents were estimated.
    freqs - ndarray of floats
        Most prominent frequency detected for corresponding timestamp
        values.
    """
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = window_start + N if len(x_n) >= N else len(x_n)
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n = N)  # Calculate one-sided DFT
        X_m[0] = 0  # Set the DC component to 0
        m_peak = np.argmax(np.abs(X_m))  # Find the index of the highest peak in 'X_m'
        freqs.append(m_peak/N*fs)  # Convert frequency index to wave frequency 'f' in hertz
        timestamps.append(window_end/fs)
    return timestamps, freqs