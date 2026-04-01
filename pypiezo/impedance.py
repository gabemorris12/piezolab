from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares


class Impedance:
    def __init__(self, frequency: np.ndarray, impedance_magnitude: np.ndarray, impedance_phase_deg: np.ndarray):
        self.frequency = frequency
        self.impedance_magnitude = impedance_magnitude
        self.impedance_phase_deg = impedance_phase_deg

        self.impedance_complex = self.impedance_magnitude*np.exp(1j*np.deg2rad(self.impedance_phase_deg))
        self.angular_frequency = 2*np.pi*self.frequency
        self.admittance_complex = 1/self.impedance_complex
        self.real_impedance = np.real(self.impedance_complex)
        self.real_admittance = np.real(self.admittance_complex)
        self.admittance_phase_deg = -self.impedance_phase_deg

        self.resonant_frequency = None  # fR
        self.anti_resonant_frequency = None  # fA
        self.motional_resistance = None  # R1
        self.static_capacitance = None  # C0
        self.resonant_quality_factor = None  # Qm
        self.motional_capacitance = None  # C1
        self.motional_inductance = None  # L1

        self.resolve_resonance()

    def resolve_resonance(self):
        admittance_arg_max = np.argmax(self.real_admittance)
        max_real_admittance = self.real_admittance[admittance_arg_max]
        self.resonant_frequency = self.frequency[admittance_arg_max]
        self.anti_resonant_frequency = self.frequency[np.argmax(self.real_impedance)]

        # R1 and C0 calculations
        self.motional_resistance = 1/np.max(self.real_admittance)
        Y_imag_at_resonance = np.imag(self.admittance_complex[admittance_arg_max])
        angular_resonant_frequency = self.angular_frequency[admittance_arg_max]
        self.static_capacitance = abs(Y_imag_at_resonance/angular_resonant_frequency)

        # Qm calculation
        half_Y_r_max = max_real_admittance/2
        f1 = np.interp(half_Y_r_max, self.real_admittance[:admittance_arg_max], self.frequency[:admittance_arg_max])
        f2 = np.interp(-half_Y_r_max, -self.real_admittance[admittance_arg_max:], self.frequency[admittance_arg_max:])
        del_f = f2 - f1
        self.resonant_quality_factor = self.resonant_frequency/del_f

        # C1 and L1 calculations
        # self.motional_capacitance = 1/(angular_resonant_frequency*self.motional_resistance*self.resonant_quality_factor)
        self.motional_capacitance = self.static_capacitance*(
                (self.anti_resonant_frequency/self.resonant_frequency)**2 - 1)
        self.motional_inductance = 1/(self.motional_capacitance*angular_resonant_frequency**2)

    def plot_real(self) -> tuple[plt.Figure, Any, Any]:
        if self.resonant_frequency is None or self.anti_resonant_frequency is None:
            self.resolve_resonance()

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        impedance_color = color_cycle[1]
        resonant_color = color_cycle[2]
        anti_resonant_color = color_cycle[3]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        line1, = ax.plot(self.frequency, self.real_admittance, label='Real Admittance (S)', zorder=2)
        line2, = ax2.plot(self.frequency, self.real_impedance, color=impedance_color,
                          label=r'Real Impedance ($\Omega$)', zorder=2)
        # noinspection PyTypeChecker
        p1 = ax.scatter(self.resonant_frequency, np.max(self.real_admittance), color=resonant_color, zorder=3,
                        label=f'$f_R={int(self.resonant_frequency)}$')
        p2 = ax2.scatter(self.anti_resonant_frequency, np.max(self.real_impedance), color=anti_resonant_color, zorder=3,
                         label=f'$f_A={int(self.anti_resonant_frequency)}$')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Real Admittance (S)')
        ax2.set_ylabel(r'Real Impedance ($\Omega$)')
        ax2.grid(visible=False)
        ax.legend(handles=[line1, line2, p1, p2])

        return fig, ax, ax2

    def plot_impedance(self) -> tuple[plt.Figure, Any, Any]:
        if self.resonant_frequency is None or self.anti_resonant_frequency is None:
            self.resolve_resonance()

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        phase_color = color_cycle[1]
        resonant_colors = color_cycle[2]

        parameter_response = self.equivalent_impedance(1j*self.angular_frequency)
        parameter_mag = np.abs(parameter_response)

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.set_xscale('log')
        ax.set_yscale('log')

        line1, = ax.plot(self.frequency, self.impedance_magnitude, label=r'$|Z|$ ($\Omega$)', zorder=2)
        line2, = ax2.plot(self.frequency, self.impedance_phase_deg, color=phase_color, label=r'Phase (deg)',
                          zorder=2)
        line3, = ax.plot(self.frequency, parameter_mag, ls='--', zorder=2, label='Equivalent Circuit $|Z|$')
        # noinspection PyTypeChecker
        ax.axvline(x=self.resonant_frequency, color=resonant_colors, ls='--')
        ax.axvline(x=self.anti_resonant_frequency, color=resonant_colors, ls='--')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'$|Z|$ ($\Omega$)')
        ax2.set_ylabel('Phase (deg)')
        ax2.grid(visible=False)
        ax.legend(handles=[line1, line2, line3])

        return fig, ax, ax2

    def equivalent_impedance(self, s: complex | np.ndarray) -> complex | np.ndarray:
        return _equivalent_impedance(s, self.static_capacitance, self.motional_capacitance, self.motional_resistance,
                                     self.motional_inductance)

    def summarize(self):
        BVD_model = """
            |------|| C0 ------|
            |                  |
---o--------|                  |--------o---
            |                  |
            |---R1---L1---C1---|
"""

        print("BVD Model Parameters:")
        print(BVD_model)
        print(f'fR (Resonant Frequency): {self.resonant_frequency:.2f} Hz')
        print(f'fA (Anti-Resonant Frequency): {self.anti_resonant_frequency:.2f} Hz')
        print(f'R1 (Motional Resistance): {self.motional_resistance:.2f} Ω')
        print(f'C0 (Static Capacitance): {self.static_capacitance:.2e} F')
        print(f'Qm (Resonant Quality Factor): {self.resonant_quality_factor:.2f}')
        print(f'C1 (Motional Capacitance): {self.motional_capacitance:.2e} F')
        print(f'L1 (Motional Inductance): {self.motional_inductance:.2e} H')

    def curve_fit(self):
        if self.resonant_frequency is None or self.anti_resonant_frequency is None:
            self.resolve_resonance()

        ydata = np.concatenate((self.real_impedance, np.imag(self.impedance_complex)))
        popt, *_ = curve_fit(self._curve_fit_method, self.angular_frequency, ydata,
                             p0=[self.static_capacitance, self.motional_capacitance, self.motional_resistance,
                                 self.motional_inductance])
        self.static_capacitance, self.motional_capacitance, self.motional_resistance, self.motional_inductance = popt

    def least_squares_fit(self, resonance_weight: float = 5.0, resonance_width: float | None = None,
                          frequency_weights: np.ndarray | None = None):
        if self.resonant_frequency is None or self.anti_resonant_frequency is None:
            self.resolve_resonance()

        p0 = np.array([
            self.static_capacitance,
            self.motional_capacitance,
            self.motional_resistance,
            self.motional_inductance,
        ], dtype=float)
        result = least_squares(
            self._least_squares_residuals,
            x0=p0,
            bounds=(0, np.inf),
            args=(resonance_weight, resonance_width, frequency_weights),
        )
        self.static_capacitance, self.motional_capacitance, self.motional_resistance, self.motional_inductance = \
            result.x

        return result

    @staticmethod
    def _curve_fit_method(omega, C0, C1, R1, L1):
        impedance = _equivalent_impedance(1j*omega, C0, C1, R1, L1)
        real_impedance = np.real(impedance)
        imag_impedance = np.imag(impedance)
        return np.concatenate((real_impedance, imag_impedance))

    def _least_squares_residuals(self, params: np.ndarray, resonance_weight: float, resonance_width: float | None,
                                 frequency_weights: np.ndarray | None) -> np.ndarray:
        model = self._curve_fit_method(self.angular_frequency, *params)
        observed = np.concatenate((self.real_impedance, np.imag(self.impedance_complex)))
        weights = self._frequency_weights(resonance_weight, resonance_width, frequency_weights)
        return (model - observed)*np.concatenate((weights, weights))

    def _frequency_weights(self, resonance_weight: float, resonance_width: float | None,
                           frequency_weights: np.ndarray | None) -> np.ndarray:
        if frequency_weights is not None:
            weights = np.asarray(frequency_weights, dtype=float)
            if weights.shape != self.frequency.shape:
                raise ValueError("frequency_weights must match the frequency array shape.")
            return weights

        if resonance_width is None:
            resonance_width = max(abs(self.anti_resonant_frequency - self.resonant_frequency)*2*np.pi/4,
                                  np.finfo(float).eps)
        elif resonance_width <= 0:
            raise ValueError("resonance_width must be positive.")

        if resonance_weight < 1:
            raise ValueError("resonance_weight must be at least 1.")

        scaled_distance = 2*np.pi*(self.frequency - self.resonant_frequency)/resonance_width
        return 1 + (resonance_weight - 1)*np.exp(-0.5*scaled_distance**2)


def _equivalent_impedance(s, C0, C1, R1, L1):
    num = C1*L1*s**2 + C1*R1*s + 1
    den = C0*C1*L1*s**3 + C0*C1*R1*s**2 + (C0 + C1)*s
    return num/den
