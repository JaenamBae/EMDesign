from EM.CouplingCoefficientInterpolator import CouplingCoefficientInterpolator2


file_path = "FluxLinkage_IPMSM.csv"

interpolator = CouplingCoefficientInterpolator2(file_path)

ratio = 8/12
tau_so = 0.2
harmonic = 3
tau_g = 0.012

k_wc = interpolator.interpolate(harmonic, tau_so, tau_g, ratio)
print(k_wc)

interpolator.plot_harmonics_3d([5,7,9], 0.012)