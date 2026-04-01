from importlib.resources import files

import numpy as np

PATH = files("pypiezo").joinpath('data')

SAMPLE_OPTIONS = (
    "clean_resonant_data",
    "clean_sweep_data",
    'messy_sweep_data',
    'messy_resonant_data'
)


# noinspection PyTypeChecker
def get_data(sample_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_name not in SAMPLE_OPTIONS:
        raise ValueError(f"Invalid sample name. Available options: {SAMPLE_OPTIONS}")

    data = np.load(str(PATH.joinpath(f"{sample_name}.npy")))
    frequencies = np.real(data[:, 0])
    complex_impedance = data[:, 1]
    impedance = np.abs(complex_impedance)
    phase = np.angle(complex_impedance, deg=True)
    return frequencies, impedance, phase
