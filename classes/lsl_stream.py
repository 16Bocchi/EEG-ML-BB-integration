import threading
import warnings
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop

DEFAULT_SAMPLE_RATE = 1000


class LSLStream:
    def __init__(self, stream_to_find, buffer_duration_s, max_chunk_len) -> None:

        # Stream
        self.stream_to_find = stream_to_find
        self.stream_info = None
        self.stream_inlet = None
        self.max_chunk_len = max_chunk_len

        # Buffer
        self._buffer = deque()
        self._buffer_lock = threading.Lock()
        self.buffer_duration_s = buffer_duration_s

        # Metadata
        self.channel_count = None
        self.nominal_sample_rate = None
        self.name = None
        self.type = None

        # Threading
        self._thread = None
        self._stop_event = threading.Event()

        # Callback
        self.chunk_callback = None

    def search_and_open(self, timeout_s):
        # Create stream
        property, value = self.stream_to_find
        info = resolve_byprop(prop=property, value=value, timeout=timeout_s)
        if not info:
            raise RuntimeError(f"No LSL stream found for: {property}, {value}")

        self.stream_info = info[0]
        self.stream_inlet = StreamInlet(
            self.stream_info, max_chunklen=self.max_chunk_len
        )

        # Metadata
        try:
            self.channel_count = int(self.stream_info.channel_count())
        except Exception:
            self.channel_count = None
            warnings.warn("Failed to extract channel count, defaulting to None")

        try:
            self.nominal_sample_rate = float(self.stream_info.nominal_srate())
        except Exception:
            self.nominal_sample_rate = None
            warnings.warn("Failed to extract nominal sample rate, defaulting to None")

        try:
            self.name = self.stream_info.name()
            self.type = self.stream_info.type()
        except Exception:
            self.name = None
            self.type = None
            warnings.warn("Failed to extract name and type, defaulting to None")

        if self.nominal_sample_rate and self.nominal_sample_rate > 0:
            max_samples = int(self.nominal_sample_rate * self.buffer_duration_s)
        else:
            warnings.warn(
                "Falling back to default sample rate for max samples estimate"
            )
            max_samples = int(DEFAULT_SAMPLE_RATE * self.buffer_duration_s)
        self._max_samples_estimate = max_samples

    def _looped_pull(self, pull_timeout_s):
        if self.stream_inlet is None:
            raise RuntimeError("Inlet not opened...")

        while not self._stop_event.is_set():
            try:
                samples_list, timestamps_list = self.stream_inlet.pull_chunk(
                    timeout=pull_timeout_s, max_samples=self.max_chunk_len
                )
            except Exception:
                samples_list, timestamps_list = [], []

            if not samples_list:
                continue

            samples = np.asarray(samples_list, dtype=float)
            timestamps = np.asarray(timestamps_list, dtype=float)

            with self._buffer_lock:
                self._buffer.append((samples, timestamps))
                self._enforce_buffer_limits_locked()

            if self.chunk_callback:
                try:
                    self.chunk_callback(samples, timestamps)
                except Exception as e:
                    warnings.warn(f"[LSLStreamInlet: {self.name}] callback error: {e}")

    def _enforce_buffer_limits_locked(self):
        total_samples = sum(chunk[0].shape[0] for chunk in self._buffer)

        if self._buffer:
            try:
                first_timestamp = self._buffer[0][1][0]
                last_timestamp = self._buffer[-1][1][-1]
                span = last_timestamp - first_timestamp
            except Exception:
                span = None
        else:
            span = None

        if span is not None and span > self.buffer_duration_s:
            while self._buffer and (
                self._buffer[-1][1][-1] - self._buffer[0][1][0] > self.buffer_duration_s
            ):
                self._buffer.popleft()
        else:
            if total_samples > 2 * self._max_samples_estimate:
                num_to_pop = max(1, len(self._buffer) // 2)
                for _ in range(num_to_pop):
                    self._buffer.popleft()

    def start(self, pull_timeout_s, search_timeout_s):
        if self.stream_inlet is None:
            self.search_and_open(search_timeout_s)

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._looped_pull, args=(pull_timeout_s,), daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout_s):
        self._stop_event.clear()
        if self._thread:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None

    def clear_buffer(self):
        with self._buffer_lock:
            self._buffer.clear()

    def register_callback(self, cb):
        self.chunk_callback = cb

    def info_summary(self) -> dict:
        return {
            "stream_to_find": self.stream_to_find,
            "name": self.name,
            "type": self.type,
            "channel_count": self.channel_count,
            "nominal_ssample_rate": self.nominal_sample_rate,
        }
