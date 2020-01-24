import matplotlib.pyplot as plt
#info.redefine = {'Lambda_over_theoritical_Lambda': 'Lambda_over_theoritical_Lambda*8.2206'}
#info.redefine = {'omega_cdm':'(omega_b/100+omega_cdm)*10000/H0/H0','sigma8':'(sigma8*(((omega_b/100+omega_cdm)*10000/H0/H0)/0.27)**0.3-0.782)/0.010'}
info.to_change = {'T_cmb':'T_0','100*theta_s':r'100$\theta_s$'}
#info.to_change = {'T_cmb':'T_0','100*theta_s':r'100$\theta_s$','omega_cdm':'Omega_m','sigma8':'r$\sigma_{\rm SZ}$'}
#info.to_plot = ['omega_b','100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb',r'$p_{\mathrm{ann}}$']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
info.to_plot = ['T_0','tau_reio','omega_cdm',r'100$\theta_s$', 'ln10^{10}A_s','n_s','omega_b','Omega_m','H0','Omega_m','r$\sigma_{\rm SZ}$']

#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['T0 = 2.7255 K','T0 = 2.6 K']
#info.legendnames = ['without Lensing','with Lensing']
info.legendnames = ['liteCORE 80','liteCORE 120','CORE+']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 3
info.ticksize = 20
info.fontsize = 25
info.line_width = 2
info.decimals = 3
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
info.cm = [
        (0.,      0.,      0.,      1.),
        (0.30235, 0.15039, 0.74804, 1.),
        (0.99843, 0.25392, 0.14765, 1.),
        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
info.cmaps = [plt.cm.gray_r, plt.cm.Purples, plt.cm.Reds_r, plt.cm.Greens]
info.alphas = [1.0, 0.8, 0.6, 0.4]
