import matplotlib.pyplot as plt

from piezolab import Impedance, get_data

# plt.style.use('../maroon_py.mplstyle')

res_fr, res_impedance, res_phase = get_data('clean_resonant_data')

impedance = Impedance(res_fr, res_impedance, res_phase)
impedance.least_squares_fit(resonance_weight=100)
impedance.plot_real()
impedance.plot_impedance()
impedance.summarize()
plt.show()
