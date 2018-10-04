import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(log10decay)'}
#info.redefine = {'decay':'1e-26*decay'}
#info.to_change = {'log10decay':'Decay'}
info.to_change = {'decay':r'$\Gamma_{\mathrm{eff}}$','100*theta_s':r'$100~\theta_s$','lambda_duspis_et_al':r'$\lambda$','z_reio':r'$z_{\rm reio}$'}
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
info.to_plot = ['tau_reio',r'$\Gamma_{\mathrm{eff}}$',r'$z_{\rm reio}$',r'$\lambda$']
#info.to_plot = ['omega_b','omega_cdm',r'$100~\theta_s$', 'ln10^{10}A_s','n_s','tau_reio',r'$\Gamma_{\mathrm{eff}}$','z_reio','tau_dcdm',r'$\lambda$','log10decay','sigma8','H0',r'$z_{\rm reio}$']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
#info.to_plot = [r'$\sum m_\nu$','tau_reio']
#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['LCDM+Tcmb Hyrec','LCDM Hyrec','LCDM recfast']
#info.legendnames = ['camb-like reionization','asymmetric reionization']
#info.legendnames = ['LCDM recfast','LCDM hyrec']
info.legendsize=20
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.legendnames = ['LiteCORE80','CORE-M5','CORE+']
info.legendnames = ['Planck','CORE-M5']
#info.legendnames = ['1e12s','1e14s','1e16s','1e20s']
#info.legendnames = ['Low-P','simlow']

info.ticknumber = 3
info.ticksize = 15
info.line_width = 2
info.fontsize = 20
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
#info.bins=15
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
	(0.,1.,0.,1.),
        (0.,	  0.,	   0.,      1.),
        (0.99843, 0.25392, 0.14765, 1.),
        (0, 0, 1., 1.)]
	
    # Define colormaps for the contour plots
info.cmaps = [plt.cm.Greens_r,plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues]
info.alphas = [1.0, 0.8, 0.6, 0.4]

