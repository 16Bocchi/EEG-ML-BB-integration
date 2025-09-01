import matplotlib.pyplot as plt
import numpy as np

from classes.ETP import autocorrect, compute_bias, signal_processing
from classes.ETP.config import ETPConfig


def generate_alpha_signal(fs=500, duration=5, alpha_freq=10):
    """Generate a synthetic alpha rhythm (10 Hz) with noise."""
    time = np.arange(0, duration, 1 / fs)
    signal = np.sin(2 * np.pi * alpha_freq * time)
    rng = np.random.default_rng()
    signal += 0.3 * rng.standard_normal(len(time))  # Add noise
    return signal, time


def main():
    # --- Generate synthetic data ---
    fs = 500
    duration = 10  # Increased to 10s to ensure enough samples
    signal, time = generate_alpha_signal(fs=fs, duration=duration)

    # Simulate multi-channel EEG (replicate signal across 3 electrodes)
    data_array = np.vstack([signal, signal, signal])  # (3, n_samples)

    # --- ETP Config ---
    config = ETPConfig(
        target_bounds_freq_hz=(8, 13),
        sample_rate=fs,
        electrodes_of_interest=["O1", "Oz", "O2"],
        data_length_s=duration,
        training_length_s=3,  # at least a few seconds for training
        max_trials=50,
        edge_factor=3.0,
    )

    # --- Run Processing Pipeline ---
    processor = signal_processing.SignalProcessor(config)
    autocorrection = autocorrect.Autocorrect(signal_processor=processor)

    actual_phase, full_cycle, best_bias, history, trial_history = autocorrection.run(
        data_array
    )

    # --- Plot Results ---
    filtered_signal = processor.bandpass_filter(signal)
    alpha_phase = np.angle(processor.compute_phase(filtered_signal))

    plt.figure(figsize=(14, 6))
    plt.plot(time, alpha_phase, label="Alpha phase")
    plt.plot(time, filtered_signal, alpha=0.3, label="Filtered EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase / Amplitude")
    plt.title("Synthetic Alpha Signal Processed by Pipeline")
    plt.legend()
    # plt.show()

    # --- Print Results ---
    print(f"Full cycle length: {full_cycle} samples")
    print(f"Best bias found: {best_bias}")
    print(f"Phase samples: {actual_phase[:10]}")  # preview first 10 phases

    next_target, _, _ = compute_bias.compute_bias(data_array, best_bias, processor)

    plt.figure(figsize=(14, 6))
    plt.plot(time, alpha_phase, label="Alpha phase")
    plt.plot(time, filtered_signal, alpha=0.3, label="Filtered EEG")
    plt.scatter(
        next_target / fs,
        alpha_phase[next_target.astype(int)],
        color="red",
        marker="x",
        s=100,
        label="Predicted peaks",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Synthetic Alpha Signal Processed by Pipeline")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # --- Bias Evolution ---
    plt.figure(figsize=(10, 5))
    plt.plot(trial_history, history, marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Bias (samples)")
    plt.title("Bias Evolution Across Trials")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
