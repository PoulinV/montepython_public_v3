import matplotlib.pyplot as plt
#info.redefine = {'Lambda_over_theoritical_Lambda': 'Lambda_over_theoritical_Lambda*8.2206'}
info.to_change = {'Lambda_over_theoritical_Lambda':r'$A_{2s1s}/8.2206$','100*theta_s':r'$100~\theta_s$'}

#info.to_plot = ['omega_b','100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb',r'$p_{\mathrm{ann}}$']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
info.to_plot = [r'$A_{2s1s}/8.2206$','tau_reio','omega_cdm',r'$100~\theta_s$', 'ln10^{10}A_s','n_s','omega_b','Omega_m','sigma8','H0']

#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['LCDM+Tcmb Hyrec','LCDM Hyrec','LCDM recfast']
#info.legendnames = ['core','core+desi']
info.legendnames = ['liteCORE80','CORE-M5','CORE+']
info.legendsize=20
info.fontsize=20
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 3
info.ticksize = 15
info.line_width = 2
info.decimal = 4
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=20
#info.cm = [
	#(0.30235, 0.15039, 0.74804, 1.),
#        (0.05,0.05,1.,1.),
 #       (0.99843, 0.25392, 0.14765, 1.),
  #      (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots#
#info.cmaps = [plt.cm.Blues, plt.cm.Reds,plt.cm.Yellows]
#info.cmaps = [plt.cm.Purples, plt.cm.Reds_r, plt.cm.Greens]
#info.alphas = [1.0, 0.6,0.4]

    # Define colormaps for the contour plots
info.cm = [
	(0.,1.,0.,1.),
        (0.,	  0.,	   0.,      1.),
        (0.99843, 0.25392, 0.14765, 1.),
        (0, 0, 1., 1.),
        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
info.cmaps = [plt.cm.Greens,plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues,plt.cm.Greens]
info.alphas = [1.0, 0.8, 0.6, 0.4]

