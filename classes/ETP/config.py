from dataclasses import dataclass


@dataclass
class ETPConfig:
    target_bounds_freq_hz: tuple[float, float]
    sample_rate: int
    electrodes_of_interest: list[str]
    data_length_s: int
    training_length_s: int
    max_trials: int
    edge_factor: float
