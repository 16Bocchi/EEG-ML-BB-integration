import threading
import time

from classes.lsl_stream import LSLStream
from classes.mock_lsl import MockLSLStreamer
from classes.xdf_loader import EEGXDFLoader


def run_streamer():
    loader = EEGXDFLoader("data/P001_S001.xdf")
    loader.load()
    loader.select_stream(eeg_name="LiveAmpSN-055606-0358")
    streamer = MockLSLStreamer(loader)
    streamer.start(speed_factor=1.0)


def run_receiver():
    def on_new_chunk(samples, timestamps):
        print(f"Received chunk: {samples.shape} at {timestamps[0]}")

    receiver = LSLStream(("name", "MockEEG"), buffer_duration_s=5, max_chunk_len=32)
    receiver.register_callback(on_new_chunk)
    receiver.start(pull_timeout_s=0.1, search_timeout_s=5)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    threading.Thread(target=run_streamer, daemon=True).start()
    run_receiver()
