import matplotlib.pyplot as plt
#info.redefine = {'Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'Gamma_dcdm':'f_dcdm Gamma_dcdm' }
info.to_change={'alpha_asymmetric_planck_16':r'$\alpha$','z_end_asymmetric_planck_16':r'z_{\rm end, true}','z_99_percent':r'z_{\rm end}','z_10_percent':r'z_{\rm beg}','z_50_percent':r'z_{\rm reio}','duration_of_reionization':r'$\Delta z_{\rm reio}$','100*theta_s':r'$100~\theta_s$'}
info.legendnames = ['LiteCORE80','CORE-M5','CORE+']
#info.legendnames = ['instanteneous reionization','asymmetric reionization']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 3
info.ticksize = 15
info.decimals=3
info.fontsize = 25
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}',r'$\alpha$',r'z_{\rm end, true}']
#info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=15
#info.cm = [
#	(0.,	  0.,	   0.,      1.),
#        (0.99843, 0.25392, 0.14765, 1.),
#        (0, 0, 1., 1.),
#        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
#info.cmaps = [plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues,plt.cm.Greens]
#info.alphas = [1.0, 0.8, 0.6, 0.4]
info.cm = [
	(0.,1.,0.,1.),
        (0.,	  0.,	   0.,      1.),
        (0.99843, 0.25392, 0.14765, 1.),
        (0, 0, 1., 1.)]

    # Define colormaps for the contour plots
info.cmaps = [plt.cm.Greens_r,plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues]
info.alphas = [1.0, 0.8, 0.6, 0.4]