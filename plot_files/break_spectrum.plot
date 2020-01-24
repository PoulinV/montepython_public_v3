import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(log10decay)'}
#info.redefine = {'decay':'1e-26*decay'}
#info.to_change = {'log10decay':'Decay'}
#info.to_change = {'decay':r'$\Gamma_{\mathrm{eff}}$','100*theta_s':r'$100~\theta_s$','lambda_duspis_et_al':r'$\lambda$','z_reio':r'$z_{\rm reio}$'}
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['tau_reio',r'$\Gamma_{\mathrm{eff}}$',r'$z_{\rm reio}$',r'$\lambda$']
info.redefine = {'N_ur':'N_ur+1.0132-3.046'}
#info.redefine = {'omega_ini_dcdm':'omega_ini_dcdm/(omega_ini_dcdm+omega_cdm)','omega_cdm':'omega_ini_dcdm+omega_cdm','N_ur':'N_ur+1.0132-3.046'}
info.to_change = {'N_ur':r'$\Delta N_{\rm eff}$','H0':'H_0','k_pivot_ms':r'$k_{m_s}$'}
#info.to_change = {'log10_Gamma_dcdm2bar':r'${\rm log}_{10}\Gamma_{\rm dcdm}$'}
#info.to_plot = ['log10_Gamma_dcdm2bar','omega_ini_dcdm','omega_cdm','H0']
#info.to_plot = [r'${\rm log}_{10}\Gamma_{\rm dcdm}$',r'$f_{\rm dcdm}$',r'$\omega_{\rm dmtot}$','H_0','omega_b','n_s','A_s','tau_reio','omega_ini_dcdm','omega_cdm',r'$\Delta N_{\rm eff}$']
info.to_plot = ['n_s','m_s',r'$k_{m_s}$','H_0',r'$\Delta N_{\rm eff}$']
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
#info.legendnames = ['DCDM','DCDM+Neff']
#info.legendnames = ['LCDM','DCDM-to-DR','DCDM-to-bar','DCDM-to-DR+Neff']
#info.legendnames = ['DCDM-to-DR','DCDM-to-DR+Neff']
#info.legendnames = ['DCDM-to-bar','DCDM-to-DR+Neff','Neff']
info.legendnames = ['LCDM','Neff+break']
#info.legendnames = ['Planck','CORE-M5']
#info.legendnames = ['1e12s','1e14s','1e16s','1e20s']
#info.legendnames = ['Low-P','simlow']

info.ticknumber = 3
info.ticksize = 15
info.line_width = 2
info.fontsize = 20
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=15
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
info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']
