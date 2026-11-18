from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares


# noinspection pep8-naming
class Impedance:
    def __init__(self, frequency: np.ndarray, impedance_magnitude: np.ndarray, impedance_phase_deg: np.ndarray):
        """
        Container for impedance data and BVD model parameters

        :param frequency: Frequency array in Hz
        :param impedance_magnitude: Magnitude of impedance in Ohms
        :param impedance_phase_deg: Phase of impedance in degrees

        Instance attributes:
        - impedance_complex: Complex impedance calculated from magnitude and phase
        - angular_frequency: Angular frequency array in rad/s
        - admittance_complex: Complex admittance calculated as 1/impedance
        - real_impedance: Real part of the impedance
        - real_admittance: Real part of the admittance
        - admittance_phase_deg: Phase of the admittance in degrees (negative of impedance phase
        - resonant_frequency: Resonant frequency (fR) in Hz
        - anti_resonant_frequency: Anti-resonant frequency (fA) in Hz
        - motional_resistance: Motional resistance (R1) in Ohms
        - static_capacitance: Static capacitance (C0) in Farads
        - resonant_quality_factor: Resonant quality factor (Qm) dimensionless
        - motional_capacitance: Motional capacitance (C1) in Farads
        - motional_inductance: Motional inductance (L1) in Henries

        Instance method:
        - resolve_resonance: Calculate resonant and anti-resonant frequencies, and BVD parameters. Called automatically
                             during initialization.
        - plot_real: Plot real admittance and real impedance vs frequency, highlighting resonant and anti-resonant
                     frequencies.
        - plot_impedance: Plot impedance magnitude and phase vs frequency, with optional equivalent circuit overlay
        - equivalent_impedance: Calculate the impedance of the equivalent BVD circuit for a given complex frequency s
        - summarize: Print a summary of the BVD model parameters
        - curve_fit: Fit the BVD model parameters to the impedance data using non-linear least squares optimization.
        - least_squares_fit: Fit the BVD model parameters using a custom least squares optimization with optional
                             weighting around the resonance.
        """
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

        self._off_res_cap = None  # Optional: Off-resonance capacitance for constrained fitting

        self.resolve_resonance()

    # noinspection unsupported-operator
    def resolve_resonance(self):
        """
        Evaluate the resonant and anti-resonant frequencies from the real admittance and impedance data.
        """
        admittance_arg_max = np.argmax(self.real_admittance)
        max_real_admittance = self.real_admittance[admittance_arg_max]
        self.resonant_frequency = self.frequency[admittance_arg_max]
        self.anti_resonant_frequency = self.frequency[np.argmax(self.real_impedance)]

        # R1 and C0 calculations
        self.motional_resistance = 1/np.max(self.real_admittance)
        y_imag_at_resonance = np.imag(self.admittance_complex[admittance_arg_max])
        angular_resonant_frequency = self.angular_frequency[admittance_arg_max]
        self.static_capacitance = abs(y_imag_at_resonance/angular_resonant_frequency)

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
        """
        Generate a plot of the real admittance and real impedance vs frequency, highlighting the resonant and
        anti-resonant frequencies.

        :return: A tuple containing the matplotlib Figure and the two Axes objects for the admittance and impedance
                 plots.
        """
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
        """
        Generate a plot of the impedance magnitude and phase vs frequency with the BVD equivalent circuit response
        overlaid.

        :return: A tuple containing the matplotlib Figure and the two Axes objects for the impedance magnitude and phase
                 plots.
        """
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
        ax.axvline(x=self.resonant_frequency, color=resonant_colors, ls='--')
        ax.axvline(x=self.anti_resonant_frequency, color=resonant_colors, ls='--')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'$|Z|$ ($\Omega$)')
        ax2.set_ylabel('Phase (deg)')
        ax2.grid(visible=False)
        ax.legend(handles=[line1, line2, line3])

        return fig, ax, ax2

    def equivalent_impedance(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Calculate the impedance of the equivalent BVD circuit for a given complex frequency s.

        :param s: Complex frequency (s = j*omega) at which to evaluate the impedance.
        :return: The complex impedance of the equivalent BVD circuit at the given frequency.
        """
        return _equivalent_impedance(s, self.static_capacitance, self.motional_capacitance, self.motional_resistance,
                                     self.motional_inductance)

    # noinspection string-format
    def summarize(self):
        """
        Print a summary of the BVD model parameters in a clear and organized format. The circuit is shown below

                    |------|| C0 ------|
                    |                  |
        ---o--------|                  |--------o---
                    |                  |
                    |---R1---L1---C1---|
        """
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

    def curve_fit(self, off_res_cap: float | bool | None = None):
        """
        Fit the BVD model parameters to the impedance data using non-linear least squares optimization. If off_res_cap
        is provided as a float, it will be used as the off-resonance capacitance to constrain the fit. If off_res_cap is
        True, the off-resonance capacitance will be estimated from the low-frequency data and used to constrain the fit.
        Ensure that the low-frequency data is sufficiently off-resonance for an accurate estimate.

        :param off_res_cap: Optional parameter to constrain the fit with an off-resonance capacitance
        """
        if self.resonant_frequency is None or self.anti_resonant_frequency is None:
            self.resolve_resonance()

        if isinstance(off_res_cap, float):
            self._off_res_cap = off_res_cap
            fit_func = self._curve_fit_method_constrain_off_resonance
            p0 = [self.static_capacitance, self.motional_resistance, self.motional_inductance]
        elif off_res_cap:
            self._find_off_resonance_capacitance()
            fit_func = self._curve_fit_method_constrain_off_resonance
            p0 = [self.static_capacitance, self.motional_resistance, self.motional_inductance]
        else:
            fit_func = self._curve_fit_method
            p0 = [self.static_capacitance, self.motional_capacitance, self.motional_resistance,
                  self.motional_inductance]

        ydata = np.concatenate((self.real_impedance, np.imag(self.impedance_complex)))
        # noinspection bad-argument-type
        popt, *_ = curve_fit(fit_func, self.angular_frequency, ydata, p0=p0)

        if not off_res_cap:
            self.static_capacitance, self.motional_capacitance, self.motional_resistance, self.motional_inductance = \
                popt
        else:
            self.static_capacitance, self.motional_resistance, self.motional_inductance = popt
            self.motional_capacitance = self._off_res_cap - self.static_capacitance

    def least_squares_fit(self, resonance_weight: float = 5.0, resonance_width: float | None = None,
                          frequency_weights: np.ndarray | None = None, off_res_cap: float | bool | None = None):
        """
        Fit the BVD model parameters using a custom least squares optimization with optional weighting around the
        resonance. The weighting can be controlled by the resonance_weight parameter, which increases the importance of
        fitting the data points near the resonant frequency. The resonance_width parameter controls the width of the
        weighting around the resonance and can be set to None to automatically determine it based on the distance
        between the resonant and anti-resonant frequencies. Alternatively, specific frequency weights can be provided
        through the frequency_weights parameter, which should be an array of the same shape as the frequency array. If
        off_res_cap is provided as a float, it will be used as the off-resonance capacitance to constrain the fit.

        :param resonance_weight: Weighting factor to increase the importance of fitting near the resonant frequency
                                 (must be at least 1).
        :param resonance_width: Width of the weighting around the resonance in rad/s. If None, it will be automatically
                                determined.
        :param frequency_weights: Optional array of weights for each frequency point. If provided, it overrides the
                                  resonance_weight and resonance_width parameters.
        :param off_res_cap: Optional parameter to constrain the fit with an off-resonance capacitance. If provided as a
                            float, it will be used directly. If True, the off-resonance capacitance will be estimated
                            from the low-frequency data and used to constrain the fit. Ensure that the low-frequency
                            data is sufficiently off-resonance for an accurate estimate.
        """
        if self.resonant_frequency is None or self.anti_resonant_frequency is None:
            self.resolve_resonance()

        if isinstance(off_res_cap, float):
            self._off_res_cap = off_res_cap
            fit_func = self._least_squares_residuals_constrain_off_resonance
            p0 = np.array([
                self.static_capacitance,
                self.motional_resistance,
                self.motional_inductance,
            ], dtype=np.float64)
        elif off_res_cap:
            self._find_off_resonance_capacitance()
            fit_func = self._least_squares_residuals_constrain_off_resonance
            p0 = np.array([
                self.static_capacitance,
                self.motional_resistance,
                self.motional_inductance,
            ], dtype=np.float64)
        else:
            fit_func = self._least_squares_residuals
            p0 = np.array([
                self.static_capacitance,
                self.motional_capacitance,
                self.motional_resistance,
                self.motional_inductance,
            ], dtype=np.float64)

        result = least_squares(
            fit_func,
            x0=p0,
            bounds=(0, np.inf),
            args=(resonance_weight, resonance_width, frequency_weights),
        )

        if not off_res_cap:
            self.static_capacitance, self.motional_capacitance, self.motional_resistance, self.motional_inductance = \
                result.x
        else:
            self.static_capacitance, self.motional_resistance, self.motional_inductance = result.x
            self.motional_capacitance = self._off_res_cap - self.static_capacitance

        return result

    @staticmethod
    def _curve_fit_method(omega, C0, C1, R1, L1):
        impedance = _equivalent_impedance(1j*omega, C0, C1, R1, L1)
        real_impedance = np.real(impedance)
        imag_impedance = np.imag(impedance)
        return np.concatenate((real_impedance, imag_impedance))

    def _curve_fit_method_constrain_off_resonance(self, omega, C0, R1, L1):
        C1 = self._off_res_cap - C0
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

    def _least_squares_residuals_constrain_off_resonance(self, params: np.ndarray, resonance_weight: float,
                                                         resonance_width: float | None,
                                                         frequency_weights: np.ndarray | None) -> np.ndarray:
        model = self._curve_fit_method_constrain_off_resonance(self.angular_frequency, *params)
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

    def _find_off_resonance_capacitance(self, frequency_width=20):
        off_res_limit = self.resonant_frequency/5*2*np.pi
        angular_frequencies = self.angular_frequency[:frequency_width]

        if angular_frequencies[-1] > off_res_limit:
            warnings.warn(f'There might not be enough off-resonance data points to accurately determine the '
                          f'off-resonance capacitance. It is recommended to check the results and maybe manually give '
                          f'the off-resonance capacitance.')

        impedances = self.impedance_magnitude[:frequency_width]
        off_res_capacitances = 1/(angular_frequencies*impedances)

        self._off_res_cap = np.mean(off_res_capacitances)


# noinspection pep8-naming
def _equivalent_impedance(s, C0, C1, R1, L1):
    num = C1*L1*s**2 + C1*R1*s + 1
    den = C0*C1*L1*s**3 + C0*C1*R1*s**2 + (C0 + C1)*s
    return num/den
