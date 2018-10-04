import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(-1*log10decay)'}
#info.redefine = {'annihilation':'1e-7*annihilation'}
#info.to_change = {'log10decay':'Decay'}
info.to_change = {'annihilation':r'$10^{7}p_{\mathrm{ann}}$'}
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
info.to_plot = ['omega_b','omega_cdm','100*theta_s', 'ln10^{10}A_s','n_s','tau_reio',r'$10^{7}p_{\mathrm{ann}}$','sigma8','H0']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
#info.to_plot = [r'$\sum m_\nu$','tau_reio']
#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['LCDM+Tcmb Hyrec','LCDM Hyrec','LCDM recfast']
#info.legendnames = ['core+simlow','core+simlow+BAO']
#info.legendnames = ['LCDM recfast','LCDM hyrec']
#info.legendsize=20
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 5
info.ticksize = 8
info.line_width = 2
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
#info.cm = [
#	(0.,1.,0.,1.),
#        (0.,      0.,      0.,      1.),
#        (0.99843, 0.25392, 0.14765, 1.),
#        (0, 0, 1., 1.),
#        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
#info.cmaps = [plt.cm.Greens,plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues,plt.cm.Greens]
#info.alphas = [1.0, 0.8, 0.6, 0.4]
