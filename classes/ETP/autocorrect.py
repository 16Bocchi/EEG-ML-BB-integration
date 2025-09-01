import numpy as np

from classes.ETP import compute_bias, signal_processing


class Autocorrect:
    def __init__(self, signal_processor: signal_processing.SignalProcessor) -> None:
        self.processor = signal_processor

    def run(self, data: np.ndarray, max_iter: int = 100):
        bias = 0
        best_bias = 0
        min_error = np.inf

        bias_history = []
        trial_numbers = []

        for i in range(max_iter):
            next_target, actual_phase, full_cycle = compute_bias.compute_bias(
                data, bias=bias, signal_processor=self.processor
            )

            mean_error = np.abs(np.nanmean(actual_phase))
            bias_history.append(bias)
            trial_numbers.append(i)

            if mean_error < min_error:
                min_error = mean_error
                best_bias = bias
            else:
                break

            # Adjust bias
            if np.nanmean(actual_phase) < 0:
                bias += 1
            else:
                bias -= 1

        # Re-run with best bias
        _, actual_phase, full_cycle = compute_bias.compute_bias(
            data, bias=best_bias, signal_processor=self.processor
        )

        return actual_phase, full_cycle, best_bias, bias_history, trial_numbers
