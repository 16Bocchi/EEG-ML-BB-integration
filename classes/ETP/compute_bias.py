import mne
import numpy as np

from classes.ETP import signal_processing


def compute_bias(
    data: np.ndarray | mne.io.RawArray,
    bias: float,
    signal_processor: signal_processing.SignalProcessor,
    debug: bool = False,  # optional debug flag
):

    config = signal_processor.config

    # Convert RawArray to numpy if needed
    if isinstance(data, mne.io.BaseRaw):
        picks = [data.ch_names.index(ch) for ch in config.electrodes_of_interest]
        data_array = data.get_data(picks=picks)
    else:
        data_array = np.array(
            [data[ch_idx, :] for ch_idx in range(len(config.electrodes_of_interest))]
        )

    # Electrode selection
    target_ch = 0
    surround_chs = list(range(1, len(config.electrodes_of_interest)))

    n_samples = data_array.shape[1]
    edge = round(
        config.sample_rate / (config.target_bounds_freq_hz[0] * config.edge_factor)
    )

    # Reference subtraction
    if surround_chs:
        ref_signal = np.mean(data_array[surround_chs, :], axis=0)
    else:
        ref_signal = np.zeros(n_samples)

    myseq = data_array[target_ch, :] - ref_signal

    # Bandpass & phase
    filtered_seq = signal_processor.bandpass_filter(myseq)
    alpha_phase = np.angle(signal_processor.compute_phase(filtered_seq))

    # Peak detection for training segment
    train_samples = int(config.sample_rate * config.training_length_s)
    locs_hi = signal_processor.find_peaks(
        filtered_seq[:train_samples],
        min_peak_dist=int(config.sample_rate / (config.target_bounds_freq_hz[1] + 3)),
    )

    # compute inter-peak intervals (ipi)
    ipi = np.diff(locs_hi)  # may be empty if <2 peaks found

    # Typical cycle length in samples - robust handling
    pks2 = np.array(ipi, dtype=float)
    # remove implausibly large intervals
    pks2[pks2 > round(config.sample_rate / 6.7)] = np.nan

    # only keep finite positive intervals for log/geom-mean
    valid = pks2[np.isfinite(pks2) & (pks2 > 0)]

    if debug:
        print("DEBUG: locs_hi (training) count:", len(locs_hi))
        print("DEBUG: ipi:", ipi)
        print("DEBUG: valid pks2:", valid)

    if valid.size == 0:
        # fallback: estimate cycle from center of target band if no valid ipi found
        center_freq = np.mean(config.target_bounds_freq_hz)
        est_cycle = config.sample_rate / center_freq
        full_cycle = int(round(est_cycle)) + int(round(bias))
        if debug:
            print(
                "WARNING: no valid peak intervals found in training segment. "
                f"Using fallback full_cycle={full_cycle} (est_cycle={est_cycle:.1f})."
            )
    else:
        # geometric mean of valid intervals (robust)
        geom_mean = float(np.exp(np.nanmean(np.log(valid))))
        full_cycle = int(round(geom_mean)) + int(round(bias))
        if debug:
            print(
                "DEBUG: geom_mean cycle samples:",
                geom_mean,
                "-> full_cycle:",
                full_cycle,
            )

    # Trial loop
    trl_num = config.max_trials
    next_target = np.full(trl_num, np.nan)
    actual_phase = np.full(trl_num, np.nan)

    for i in range(trl_num):
        start_idx = train_samples + i * 350
        stop_idx = start_idx + int(config.sample_rate / 2)

        if stop_idx > n_samples:
            break  # avoid indexing past end

        target_data = data_array[target_ch, start_idx:stop_idx]
        if surround_chs:
            ref_data = np.mean(data_array[surround_chs, start_idx:stop_idx], axis=0)
        else:
            ref_data = np.zeros_like(target_data)

        myseq2 = target_data - ref_data
        myseq2 = signal_processor.bandpass_filter(myseq2)

        locs_hi = signal_processor.find_peaks(
            myseq2[:-edge] if edge < len(myseq2) else myseq2,
            min_peak_dist=int(
                config.sample_rate / (config.target_bounds_freq_hz[1] + 1)
            ),
        )

        if debug:
            print(f"DEBUG trial {i}: found {len(locs_hi)} peaks in trial slice")

        # if no peaks in this trial slice, skip (leave NaNs)
        if len(locs_hi) == 0:
            if debug:
                print(f"Skipping trial {i}: no peaks found.")
            continue

        # Predicted target and phase
        last_peak_rel = int(locs_hi[-1])
        predicted_idx = start_idx + last_peak_rel + int(full_cycle)
        predicted_idx = int(min(predicted_idx, n_samples - 1))  # clamp to valid index
        next_target[i] = predicted_idx
        actual_phase[i] = alpha_phase[predicted_idx]

    return next_target, actual_phase, full_cycle
