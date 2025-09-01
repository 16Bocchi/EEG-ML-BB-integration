import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from classes.ETP.config import ETPConfig

BP_ORDER = 4


class SignalProcessor:
    def __init__(self, config: ETPConfig) -> None:
        self.config = config

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        low_freq, high_freq = self.config.target_bounds_freq_hz
        nyq = 0.5 * self.config.sample_rate

        low = low_freq / nyq
        high = high_freq / nyq

        # b, a are both np.ndarray
        b: np.ndarray
        a: np.ndarray
        b, a = butter(  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
            N=BP_ORDER, Wn=[low, high], btype="band"
        )
        # Apply filter along last axis
        return filtfilt(b, a, data, axis=-1)

    def compute_phase(self, data: np.ndarray) -> np.ndarray:
        analytical_signal: np.ndarray = hilbert(data, axis=-1)  #  # type: ignore
        # print(np.angle(analytical_signal)[:50])
        return np.angle(analytical_signal)

    def find_peaks(
        self,
        signal: np.ndarray,
        min_peak_dist: int = 1,
    ) -> np.ndarray:
        """
        Alternative to the findpeaks function.  This thing runs much much faster.
        It really leaves findpeaks in the dust.  It also can handle ties between
        peaks.  Findpeaks just erases both in a tie.  Shame on findpeaks.

        x is a row vector input (generally a timecourse)
        minpeakdist is the minimum desired distance between peaks
        (optional, defaults to 1)
        minpeakh is the minimum height of a peak (optional)

        (c) 2010
        Peter O'Connor
        peter<dot>ed<dot>oconnor .AT. gmail<dot>com

        Modified by Sina Shirinpour (2019, shiri008<at>umn<dot>edu) for
        Shirinpour et al., Experimental Evaluation of Methods for Real-Time EEG
        Phase-Specific Transcranial Magnetic Stimulation, 2019, bioRxiv

        Implemented in Python by Daniel Braithwaite
        """

        x = np.asarray(signal)

        locs = np.flatnonzero((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:])) + 1

        if min_peak_dist <= 1 or len(locs) <= 1:
            return locs

        while True:
            del_mask = np.diff(locs) < min_peak_dist
            if not np.any(del_mask):
                break

            pks = x[locs]

            mins = np.argmin(np.vstack([pks[:-1][del_mask], pks[1:][del_mask]]), axis=0)
            deln = np.where(del_mask)[0]
            del_idxs = deln + mins
            locs = np.delete(locs, del_idxs)
        return locs
