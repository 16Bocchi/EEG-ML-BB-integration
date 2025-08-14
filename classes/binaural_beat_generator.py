import threading
from typing import Literal

import colorednoise
import numpy as np
import sounddevice as sd
from scipy import signal


class BinauralBeatGenerator:
    """
    A real-time binaural beat generator.

    Generates two audio signals (left and right channels) with slightly different
    frequencies to create binaural beats. Supports sine, square, and sawtooth waveforms
    with smooth ramping of carrier and beat frequencies.

    Attributes:
        sample_rate (int): Audio sample rate in Hz.
        carrier_freq (float): Current carrier frequency (Hz).
        beat_freq (float): Current beat frequency (Hz).
        waveform (Literal["sine", "square", "saw"]): Current waveform type.
        amplitude (float): The amplitude (volume) of the beat
        noise_type (Literal["none", "white", "pink", "brown"]): The type of noise to add
            as a base.
        noise_amplitude (float): The amplitude of noise.
        noise_multipliter (float): How noisy the noise should be.
        target_carrier_freq (float): Target carrier frequency for ramping.
        target_beat_freq (float): Target beat frequency for ramping.
        target_amplitude (float): Target amplitude (volume) of the beat
        ramp_duration (float): Duration in seconds over which frequency ramps occur.
        ramp_progress (float): Progress in seconds of the current ramp.
        phase_left (int): Current sample offset of left chanel in waveform generation.
        phase_right (int): Current sample offset of right channel waveform generation.
        lock (threading.Lock): Lock to synchronise parameter updates.
        stream (sd.OutputStream | None): The active sounddevice output stream.
    """

    sample_rate: int
    carrier_freq: float
    beat_freq: float
    waveform: Literal["sine", "square", "saw"]
    amplitude: float
    noise_type: Literal["none", "white", "pink", "brown"]
    noise_amplitude: float
    noise_multiplier: float
    target_carrier_freq: float
    target_beat_freq: float
    target_amplitude: float
    ramp_duration: float
    ramp_progress: float
    phase_left: int
    phase_right: int
    lock: threading.Lock
    stream: sd.OutputStream | None

    def __init__(
        self,
        sample_rate: int = 44100,
        carrier_freq: float = 440,
        beat_freq: float = 10,
        waveform: Literal["sine", "square", "saw"] = "sine",
        amplitude: float = 0.2,
        ramp_duration: float = 2,
        noise_type: Literal["none", "white", "pink", "brown"] = "none",
        noise_amplitude: float = 0.1,
        noise_multiplier: float = 0.5,
    ) -> None:
        """
        Initialises the binaural beat generator.
        Args:
            sample_rate (int, default = 44100): Sample rate in Hz.
            carrier_freq (float, default = 440): Initial carrier frequency in Hz.
            beat_freq (float, default = 10): Initial beat frequency in Hz.
            waveform (Literal["sine", "square", "saw"], default = "sine): Waveform type.
            ramp_duration (float, default = 2): Duration in seconds for frequency
                ramping.
            noise_type (Literal["none", "white", "pink", "brown"], default = "none"):
                The type of noise to add as a base.
            noise_amplitude (float, default = 0.1): The amplitude of noise.
            noise_multipliter (float, default = 0.5): How noisy the noise should be.
        """
        self._sample_rate = sample_rate

        self._carrier_freq = carrier_freq
        self._beat_freq = beat_freq
        self._waveform = waveform
        self._amplitude = amplitude

        self._noise_type = noise_type
        self._noise_amplitude = noise_amplitude
        self._noise_multiplier = noise_multiplier

        self._target_carrier_freq = carrier_freq
        self._target_beat_freq = beat_freq
        self._target_amplitude = amplitude
        self._ramp_duration = ramp_duration
        self._ramp_progress = 0.0

        self._phase_left = 0
        self._phase_right = 0

        self._lock = threading.Lock()
        self._stream = None

    def __del__(self):
        self.stop()

    def start(self) -> None:
        """
        Start the audio ouput stream
        """
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=2,
            callback=self.callback,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> None:
        """
        Stop the audio output stream and close resources
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._phase_left = 0
        self._phase_right = 0
        self._ramp_progress = 0.0

    def update_frequencies(self, new_carrier_freq: float, new_beat_freq: float) -> None:
        """
        Update the target carrier and beat frequencies, starting a smooth ramp.

        Args:
            new_carrier_freq (float): New target carrier frequency in Hz.
            new_beat_freq (float): New target beat frequency in Hz.
        """
        with self._lock:
            self._target_carrier_freq = new_carrier_freq
            self._target_beat_freq = new_beat_freq
            self._ramp_progress = 0.0

    def update_waveform(self, new_waveform: Literal["sine", "square", "saw"]) -> None:
        """
        Update the waveform type used for generation.

        Args:
            new_waveform (Literal["sine", "square", "saw"]): New waveform type.
        """
        with self._lock:
            self._waveform = new_waveform

    def update_amplitude(self, new_amplitude: float) -> None:
        """Update target amplitude with ramp."""
        with self._lock:
            self._target_amplitude = max(0.0, min(new_amplitude, 1.0))  # Clamp 0..1
            self._ramp_progress = 0.0

    def update_noise(
        self,
        new_noise_type: Literal["none", "white", "pink", "brown"],
        new_noise_amplitude: float = 0.1,
        new_noise_multiplier: float = 0.5,
    ) -> None:
        with self._lock:
            self._noise_type = new_noise_type
            self._noise_amplitude = np.clip(new_noise_amplitude, 0.0, 1.0)
            self._noise_multiplier = np.clip(new_noise_multiplier, 0.0, 1.0)

    def callback(
        self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        """
        Audio callback function called by the sounddevice OutputStream.

        Generates and outputs stereo audio data for binaural beats.

        Args:
            outdata (np.ndarray): Buffer to fill with audio data.
            frames (int): Number of frames to generate.
            status (sd.CallbackFlags): Callback status flags.
        """
        if status:
            print(status)

        with self._lock:
            # Smooth ramping
            if self._ramp_duration > 0:
                alpha = min(self._ramp_progress / self._ramp_duration, 1.0)
            else:
                alpha = 1.0

            freq = (1 - alpha) * self._carrier_freq + alpha * self._target_carrier_freq
            beat = (1 - alpha) * self._beat_freq + alpha * self._target_beat_freq
            amp = (1 - alpha) * self._amplitude + alpha * self._target_amplitude

            self._ramp_progress += frames / self._sample_rate

            # Finalise ramp at end
            if alpha == 1.0:
                self._carrier_freq = self._target_carrier_freq
                self._beat_freq = self._target_beat_freq
                self._amplitude = self._target_amplitude

            waveform = self._waveform

        # Generate waveforms using phase accumulator
        left_freq = freq - beat / 2
        right_freq = freq + beat / 2

        left_phase_inc = 2 * np.pi * left_freq / self._sample_rate
        right_phase_inc = 2 * np.pi * right_freq / self._sample_rate

        phase_l = self._phase_left + left_phase_inc * np.arange(frames)
        phase_r = self._phase_right + right_phase_inc * np.arange(frames)

        self._phase_left = (phase_l[-1] + left_phase_inc) % (2 * np.pi)
        self._phase_right = (phase_r[-1] + right_phase_inc) % (2 * np.pi)

        left = self._waveform_from_phase(phase_l, waveform)
        right = self._waveform_from_phase(phase_r, waveform)

        left += self._generate_noise(self._noise_type, frames)
        right += self._generate_noise(self._noise_type, frames)

        stereo = np.stack((left, right), axis=1).astype(np.float32) * amp
        outdata[:] = stereo

    def _waveform_from_phase(self, phase: np.ndarray, waveform: str) -> np.ndarray:
        """
        Generate waveform samples based on the phase array and waveform type.

        Args:
            phase (np.ndarray): Array of phase values in radians.
            waveform (str): The type of waveform to generate. Supported values:
                'sine', 'square', 'saw'.

        Returns:
            np.ndarray: Waveform samples corresponding to the input phases.
        """
        match waveform:
            case "sine":
                return np.sin(phase)
            case "square":
                return signal.square(phase)
            case "saw":
                return signal.sawtooth(phase)
            case _:
                raise ValueError(f"Invalid waveform: {waveform}")

    def _generate_noise(self, kind: str, size: int) -> np.ndarray:
        """
        Generate colored noise using the `colorednoise` package.

        Args:
            kind (str): One of 'white', 'pink', 'brown'.
            size (int): Number of samples.

        Returns:
            np.ndarray: Noise array scaled by self._noise_multiplier.
        """
        if kind == "white":
            beta = 0  # White noise
        elif kind == "pink":
            beta = 1  # Pink noise
        elif kind == "brown":
            beta = 2  # Brownian noise
        else:
            return np.zeros(size)

        noise = colorednoise.powerlaw_psd_gaussian(beta, size)
        noise /= np.max(np.abs(noise) + 1e-8)  # Normalize
        return noise * self._noise_multiplier
