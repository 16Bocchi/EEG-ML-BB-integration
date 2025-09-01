import time

import matplotlib.pyplot as plt
import numpy as np

from classes import xdf_loader as xdfl
from classes.ETP import autocorrect, compute_bias, signal_processing
from classes.ETP.config import ETPConfig


def main():

    eeg_loader = xdfl.EEGXDFLoader("data/P001_S001.xdf")
    eeg_loader.load()
    eeg_loader.select_stream(eeg_name="LiveAmpSN-055606-0358", marker_name="CCPT-V")
    # print(eeg_loader.channel_names)
    # eeg_loader.preprocess()
    # eeg_loader.extract_events()

    raw_data = eeg_loader.raw_data  # This is an MNE RawArray
    data_array = raw_data.get_data()  # shape: (n_channels, n_times)

    config = ETPConfig(
        target_bounds_freq_hz=(8, 13),
        sample_rate=int(raw_data.info["sfreq"]),
        electrodes_of_interest=[
            "O1",
            "Oz",
            "O2",
        ],
        data_length_s=180,
        training_length_s=120,
        max_trials=255,
        edge_factor=3.0,
    )

    processor = signal_processing.SignalProcessor(config)
    start = time.time()
    autocorrection = autocorrect.Autocorrect(signal_processor=processor)
    actual_phase, full_cycle, best_bias, history, trial_history = autocorrection.run(
        data_array
    )
    print("============== time taken ================")
    print(time.time() - start)
    print("==========================================")

    # Now compute the next_target indices using compute_bias with best_bias
    next_target, _, _ = compute_bias.compute_bias(
        data_array, bias=best_bias, signal_processor=processor
    )

    print("Next target indices:", next_target[:10])
    print("Actual phase values:", actual_phase[:10])
    print("Full cycle (samples):", full_cycle)

    data_array = raw_data.get_data(picks=config.electrodes_of_interest)
    target_signal = data_array[0]  # using the first electrode as target

    # Compute filtered signal & phase (same as in compute_bias)
    filtered_signal = processor.bandpass_filter(target_signal)
    # print(filtered_signal[:50])
    alpha_phase = np.angle(processor.compute_phase(filtered_signal))

    # Time axis in seconds
    times = np.arange(len(alpha_phase)) / config.sample_rate

    plt.figure(figsize=(15, 4))
    plt.plot(times, alpha_phase, label="Alpha phase")
    plt.plot(times, filtered_signal, alpha=0.3, label="Filtered EEG")
    plt.scatter(
        next_target / config.sample_rate,
        alpha_phase[next_target.astype(int)],
        color="red",
        marker="x",
        s=100,
        label="Predicted peaks",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Alpha Phase and Predicted Target Indices")
    plt.legend()

    times = np.arange(len(filtered_signal)) / config.sample_rate

    plt.figure(figsize=(15, 4))
    plt.plot(times, filtered_signal, label="Filtered EEG (Alpha 8-13 Hz)")
    plt.scatter(
        next_target / config.sample_rate,  # x positions in seconds
        filtered_signal[next_target.astype(int)],  # y positions on waveform
        color="red",
        marker="x",
        s=100,
        label="Predicted targets",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("EEG amplitude")
    plt.title("Filtered EEG with Predicted Alpha Targets")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(trial_history, history, marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Bias (samples or radians)")
    plt.title("Bias Evolution Across Trials")
    plt.show()


if __name__ == "__main__":
    main()
