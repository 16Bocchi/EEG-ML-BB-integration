from pathlib import Path

import mne
import numpy as np
import pyxdf


class EEGXDFLoader:
    """
    Loader class for EEG and marker data from XDF files, using MNE and pyxdf.

    This class handles reading, parsing, and preprocessing EEG data and associated
    event markers. It supports automatic channel typing and montage assignment.
    """

    _filepath: Path | str
    _streams: list[dict]
    header: dict
    eeg_streams: dict[str, dict]
    marker_streams: dict[str, dict]

    _selected_eeg_stream: dict | None
    _selected_marker_stream: dict | None

    raw_data: mne.io.RawArray | None

    events: np.ndarray | None
    event_dict: dict

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialise the EEGXDFLoader with a file path to the XDF data.

        Args:
            filepath (Path | str): Path to the XDF file to be loaded.
        """
        if type(filepath) is str:
            self._filepath = Path(filepath)
        else:
            self._filepath = filepath
        self._streams = []
        self.header = {}
        self.eeg_streams = {}
        self.marker_streams = {}

        self._selected_eeg_stream = None
        self._selected_marker_stream = None

        self.raw_data = None

        self.events = None
        self.event_dict = {}

    def load(self, synchronise_clocks: bool = True):
        """
        Load the XDF file and parse its contents into EEG and marker streams.

        Args:
            synchronise_clocks (bool): Whether to synchronise clocks in the XDF file.
        """
        self._streams, self.header = pyxdf.load_xdf(
            filename=self._filepath, synchronize_clocks=synchronise_clocks
        )
        self._categorise_streams()

    def _categorise_streams(self):
        """
        Internal method to classify loaded streams into EEG and Markers.
        """
        num_streams = len(self._streams)
        for idx, stream in enumerate(self._streams):
            print(f"Extracting stream {idx+1} of {num_streams}")
            stream_type = stream["info"]["type"][0]
            stream_name = stream["info"]["name"][0]
            print(f"Stream name: {stream_name}, Stream type: {stream_type}")

            if stream_type == "EEG":
                self.eeg_streams[stream_name] = stream
                print(f"Added {stream_name} to EEG streams")
            elif stream_type == "Markers":
                self.marker_streams[stream_name] = stream
                print(f"Added {stream_name} to Marker streams")

    def list_streams(self):
        """
        Print available EEG and Marker streams.
        """
        print("EEG Streams:")
        for name in self.eeg_streams:
            print(f"  - {name}")
        print("Marker Streams:")
        for name in self.marker_streams:
            print(f"  - {name}")

    def select_stream(self, eeg_name: str, marker_name: str | None = None):
        """
        Select specific EEG and (optionally) marker stream to work with.

        Args:
            eeg_name (str): Name of the EEG stream to select.
            marker_name (str | None): Optional name of the marker stream to select.

        Raises:
            ValueError: If provided stream names do not exist.
        """
        if eeg_name not in self.eeg_streams:
            raise ValueError(f"EEG name: {eeg_name} not found.")
        self._selected_eeg_stream = self.eeg_streams[eeg_name]

        if marker_name:
            if marker_name not in self.marker_streams:
                raise ValueError(f"Marker name: {marker_name} not found.")
            self._selected_marker_stream = self.marker_streams[marker_name]
        self._convert_to_raw()

    def _convert_to_raw(self):
        """
        Internal method to convert the selected EEG stream to an MNE RawArray,
        with appropriate channel typing and montage assignment.
        """
        if not self._selected_eeg_stream:
            raise ValueError("No EEG Stream selected")

        sample_freq = float(self._selected_eeg_stream["info"]["nominal_srate"][0])

        channel_names = [
            channel["label"][0]
            for channel in self._selected_eeg_stream["info"]["desc"][0]["channels"][0][
                "channel"
            ]
        ]

        data = np.array(self._selected_eeg_stream["time_series"])
        if data.shape[1] == len(channel_names):
            data = data.T

        eeg_chans = []
        misc_chans = {}
        for ch in channel_names:
            if ch in mne.channels.make_standard_montage("standard_1005").ch_names:
                eeg_chans.append(ch)
            else:
                misc_chans[ch] = "misc"

        info = mne.create_info(
            ch_names=channel_names, ch_types="eeg", sfreq=sample_freq
        )
        self.raw_data = mne.io.RawArray(data, info)
        self.raw_data.set_channel_types(misc_chans)
        self.raw_data.set_montage("standard_1005", on_missing="ignore")

    def preprocess(self, l_freq=1.0, h_freq=55.0, reref="average"):
        """
        Apply bandpass filtering and re-referencing to the EEG data.

        Args:
            l_freq (float): Low frequency cutoff in Hz.
            h_freq (float): High frequency cutoff in Hz.
            reref (str): Re-referencing method, e.g., "average".
        """
        if self.raw_data is None:
            raise RuntimeError("Call select_streams() first.")
        self.raw_data.filter(l_freq, h_freq, fir_design="firwin")
        self.raw_data.set_eeg_reference(reref)

    def extract_events(self):
        """
        Extract events from the selected marker stream and map them to MNE format.

        Raises:
            RuntimeError: If marker or EEG stream is not selected.
        """
        if (
            self._selected_marker_stream is None
            or self.raw_data is None
            or self._selected_eeg_stream is None
        ):
            raise RuntimeError("Marker or EEG stream not selected.")
        timestamps = self._selected_marker_stream["time_stamps"]
        labels = self._selected_marker_stream["time_series"]
        sfreq = self.raw_data.info["sfreq"]
        start_time = self._selected_eeg_stream["time_stamps"][0]

        unique_labels = sorted({label[0] for label in labels})
        self.event_dict = {label: idx + 1 for idx, label in enumerate(unique_labels)}

        self.events = np.array(
            [
                [int((t - start_time) * sfreq), 0, self.event_dict[label[0]]]
                for t, label in zip(timestamps, labels)
            ]
        )

    def resample(self, new_sample_freq: float):
        """
        Resample EEG data to a new sampling frequency.

        Args:
            new_sample_freq (float): Target frequency in Hz.
        """
        if self.raw_data:
            self.raw_data.resample(new_sample_freq, npad="auto")

    def get_epochs(self, tmin=-0.2, tmax=0.8, baseline=(None, 0)):
        """
        Create epochs around events in the data.

        Args:
            tmin (float): Start time before event (in seconds).
            tmax (float): End time after event (in seconds).
            baseline (tuple): Time interval to use for baseline correction.

        Returns:
            mne.Epochs: Epochs object containing segmented EEG data.
        """
        if self.events is None:
            self.extract_events()
        return mne.Epochs(
            self.raw_data,
            self.events,
            event_id=self.event_dict,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
        )

    def plot_raw(self):
        """
        Plot the raw EEG data.
        """
        if self.raw_data:
            self.raw_data.plot()

    def plot_psd(self):
        """
        Plot the power spectral density of the EEG data.
        """
        if self.raw_data:
            self.raw_data.compute_psd().plot()

    def summary(self):
        """
        Print a summary of the current EEG data and stream selections.
        """
        print("\n=== EEGXDFLoader Summary ===")
        print(f"Loaded data from: {self._filepath}\n")
        print(f"Total streams: {len(self._streams)}")
        print(f"EEG Streams: {list(self.eeg_streams.keys())}")
        print(f"Marker Streams: {list(self.marker_streams.keys())}")
        if self.raw_data:
            print(f"\nChannels: {self.raw_data.info['nchan']}")
            print(f"Sampling Rate: {self.raw_data.info['sfreq']} Hz")
        if self._selected_marker_stream:
            print(
                f"\nSelected marker stream: "
                f"{self._selected_marker_stream['info']['name'][0]}"
            )
        if self.events is not None:
            print(
                f"\nEvents extracted: {len(self.events)} from "
                f"{len(self.event_dict)} unique labels"
            )

    @property
    def channel_names(self):
        """
        List of channel names from the raw EEG data.

        Returns:
            list[str]: Channel names if data is loaded, otherwise an empty list.
        """
        return self.raw_data.info["ch_names"] if self.raw_data else []
