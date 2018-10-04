import matplotlib.pyplot as plt
#info.redefine = {'log10Gamma_dcdm': '10**log10Gamma_dcdm'}
info.redefine = {'log10Gamma_dcdm':'log10Gamma_dcdm-2.9913998282380825'}
info.to_change = {'log10Gamma_dcdm':r'$\log_{10}(\Gamma_{\mathrm{dcdm}}$ [Gyr$^{-1}$]$)$','f_dcdm':r'$f_{\rm dcdm}$','omega_totcdm':r'$\omega_{\rm cdm}^{\rm ini}$'}
info.to_plot = [r'$f_{\rm dcdm}$', r'$\log_{10}(\Gamma_{\mathrm{dcdm}}$ [Gyr$^{-1}$]$)$',r'$\omega_{\rm cdm}^{\rm ini}$']
#info.to_plot = ['omega_b','omega_totcdm', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 7
info.ticksize = 8
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=25
#info.cmaps = [plt.cm.Blues]
#info.cm = [(0.30235, 0.15039, 0.74804, 1.),]
#info.cmaps = [plt.cm.Purples]
#info.cm = [
	#(0.30235, 0.15039, 0.74804, 1.),
#        (0.05,0.05,1.,1.),
#        (0.99843, 0.25392, 0.14765, 1.),]
#        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
info.cmaps = [plt.cm.Blues, plt.cm.Reds]
#info.cmaps = [plt.cm.Purples, plt.cm.Reds_r, plt.cm.Greens]
info.alphas = [1.0, 0.6]
#info.legendnames = ['high-$\ell$ TT','high-$\ell$ TT+TE+EE','All Experiment']
info.legendnames = ['CMB','CMB+BAO+WiggleZ']
info.line_width = 2

