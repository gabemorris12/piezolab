import matplotlib.pyplot as plt

from pypiezo import Impedance, get_data

# plt.style.use('maroon_py.mplstyle')

res_fr, res_impedance, res_phase = get_data('messy_sweep_data')

impedance = Impedance(res_fr, res_impedance, res_phase)
impedance.curve_fit(off_res_cap=True)
impedance.plot_real()
impedance.plot_impedance()
impedance.summarize()
plt.show()
