import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

from classes import xdf_loader as xdfl


def extract_freq_and_phase(raw: mne.io.Raw, channels, fmin, fmax, narrow_bw):

    fs = raw.info["sfreq"]

    if channels is None:
        channels = mne.pick_types(raw.info, eeg=True)

    x = raw.get_data(channels).mean(axis=0)

    nperseg = int(2 * fs)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    f_dom = f[mask][np.argmax(Pxx[mask])]

    low = max(1e-3, f_dom - narrow_bw / 2)
    high = min(fs / 2 - 1, f_dom + narrow_bw / 2)
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
    x_nb = filtfilt(b, a, x)

    analytic = hilbert(x_nb)
    phase = np.angle(analytic)
    amp = np.abs(analytic)

    return f_dom, phase, amp


def main():
    loader = xdfl.EEGXDFLoader("data/P001_S001_BB.xdf")
    loader.load()

    loader.list_streams()

    loader.select_stream(eeg_name="LiveAmpSN-055606-0358", marker_name="CCPT-V")

    loader.preprocess()
    loader.extract_events()

    f_dom, phase, amp = extract_freq_and_phase(loader.raw_data, ["T7"], 4.0, 40.0, 2.0)
    print("Dominant frequency:", f_dom, "Hz")
    print("Phase: ", phase[-1])


if __name__ == "__main__":
    main()
