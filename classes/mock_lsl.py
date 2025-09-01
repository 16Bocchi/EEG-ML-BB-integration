import time

from pylsl import StreamInfo, StreamOutlet

from classes.xdf_loader import EEGXDFLoader


class MockLSLStreamer:
    def __init__(self, eeg_loader: EEGXDFLoader, stream_name="MockEEG", chunk_size=32):
        """
        Args:
            eeg_loader (EEGXDFLoader): XDF loader with selected EEG stream.
            stream_name (str): Name of the LSL stream.
            chunk_size (int): Number of samples per push.
        """
        if eeg_loader.raw_data is None:
            raise ValueError("EEG loader has no raw data. Call select_stream first.")

        self.raw = eeg_loader.raw_data
        self.sfreq = self.raw.info["sfreq"]
        self.data = self.raw.get_data()  # shape: (n_channels, n_samples)
        self.data = self.data.T  # transpose to (n_samples, n_channels)

        self.chunk_size = chunk_size

        # Create LSL outlet
        self.info = StreamInfo(
            name=stream_name,
            type="EEG",
            channel_count=self.data.shape[1],
            nominal_srate=self.sfreq,
            channel_format="float32",
            source_id="mock_stream_id",
        )
        self.outlet = StreamOutlet(self.info)

    def start(self, speed_factor=1.0):
        """
        Start streaming EEG in real-time.

        Args:
            speed_factor (float): >1 for faster, <1 for slower than real-time.
        """
        n_samples = self.data.shape[0]

        for start_idx in range(0, n_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_samples)
            chunk = self.data[start_idx:end_idx]

            for sample in chunk:
                self.outlet.push_sample(sample.tolist())

            # Sleep to simulate real-time streaming
            time.sleep((len(chunk) / self.sfreq) / speed_factor)

        print("Finished streaming EEG data.")
