import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(log10decay)'}
#info.redefine = {'decay':'1e-26*decay'}
#info.to_change = {'log10decay':'Decay'}
#info.redefine = {'N_ur':'N_ur'}
info.to_change = {'100*theta_s':r'$100~\theta_s$','omega_dcdmdr':r'$\omega_{\rm dcdm+dr}$','M_tot':r'$\sum m_\nu$','Delta_Neff':r'$\Delta$ N$_{\rm eff}$','H0':'H_0','w0_fld':r'$w_0$','wa_fld':r'$w_a$','N_ur':r'$\Delta N_{\rm eff}$','sigma8':r'$\sigma_8$'}
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['omega_b','omega_cdm',r'$100~\theta_s$', 'ln10^{10}A_s','n_s','tau_reio','Omega_fld','Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta$ N$_{\rm eff}$',r'$N_{\rm eff}$']
#info.to_plot = ['Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta N_{\rm eff}$',r'$N_{\rm eff}$','cvis2_ur','ceff2_ur']
info.to_plot = ['H_0',r'$\Delta N_{\rm eff}$']
#info.to_plot = ['Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta$ N$_{\rm eff}$',r'$N_{\rm eff}$']
#info.to_plot = [r'$\omega_{\rm dcdm+dr}$',r'$\Gamma_{\rm dcdm}$','H0']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
#info.to_plot = [r'$\sum m_\nu$','tau_reio']
#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['LCDM+Tcmb Hyrec','LCDM Hyrec','LCDM recfast']
#info.legendnames = ['core+simlow','core+simlow+BAO']
#info.legendnames = ['LCDM recfast','LCDM hyrec']
info.legendsize=20
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.legendnames = ['LCDM','LCDM+mnu','LCDM+mnu+w0wa','LCDM+mnu+Nfluid+w0wa']
#info.legendnames = ['LCDM','LCDM+Neff']
#info.legendnames = ['CMB','+H0+Sz','+BAO+SDSS']
#info.legendnames = ['LCDM','LCDM+mnu','LCDM+Neff','LCDM+w0wa']
info.ticknumber = 3
info.ticksize = 14
info.line_width = 2
info.fontsize = 25
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
	(0.,1.,0.,1.),
        (0.,	  0.,	   0.,      1.),
        (0.99843, 0.25392, 0.14765, 1.),
        (0, 0, 1., 1.),
        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
info.cmaps = [plt.cm.Greens,plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues,plt.cm.Greens]
info.alphas = [1.0, 0.8, 0.6, 0.4]

# use this to control the boundaries of 1d and 2d plots
#info.force_limits = {'H_0':[65:80],'tau_reio':[0.04:0.085],r'$\sum m_\nu$':[0.258,1.05]}
# use this to customise the ticks.
#info.ticknumber = 5
#info.ticksize = 10

info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']
