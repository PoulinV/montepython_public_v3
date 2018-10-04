import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(log10decay)'}
#info.redefine = {'decay':'1e-26*decay'}
#info.to_change = {'log10decay':'Decay'}
info.to_change = {'100*theta_s':r'$100~\theta_s$','omega_dcdmdr':r'$\omega_{\rm dcdm+dr}$','M_tot':r'$\sum m_\nu$','N_ur':r'$\Delta N_{\rm fluid}$','H0':'H_0','w0_fld':r'$w_0$','wa_fld':r'$w_a$','sigma8':r'$\sigma_8$',}
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = [r'$\sigma_8$','H_0',r'$\sum m_\nu$','Omega_m']
info.to_plot = [r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta N_{\rm fluid}$','Omega_m','N_ur']
#info.to_plot = ['omega_b','omega_cdm',r'$100~\theta_s$', 'ln10^{10}A_s','n_s','tau_reio','Omega_fld','Omega_m','sigma8','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta N_{\rm fluid}$']
#info.to_plot = [r'$\omega_{\rm dcdm+dr}$',r'$\Gamma_{\rm dcdm}$','H0']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
#info.to_plot = [r'$\sum m_\nu$','tau_reio']
#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['LCDM+Tcmb Hyrec','LCDM Hyrec','LCDM recfast']
#info.legendnames = ['core+simlow','core+simlow+BAO']
#info.legendnames = ['LCDM recfast','LCDM hyrec']
info.legendsize=20
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.legendnames = ['CMB+Sz+SH0ES','All Data']
#info.legendnames = [r'$\sum m_\nu$, CMB+Sz',r'$\sum m_\nu$,CMB+Sz+BAO+SDSS',r'$\sum m_\nu,All Data$',r'$\sum m_\nu+w_0w_a$,AllData']
#info.legendnames = [r'$\Lambda$CDM',r'CPL-parametrization with $\sum m_\nu$ free',r'reconstruction with $\sum m_\nu=0.06$ eV',r'reconstruction with $\sum m_\nu$ free']
info.legendnames = [r'$\Lambda$CDM',r'$\nu_Mw$CDM+$N_{\rm fluid}$',r'$\nu_M$CDM+$N_{\rm fluid}$+$A_{\rm lens}$']
#info.legendnames = ['LCDM','LCDM+mnu','LCDM+w0wa','LCDM+mnu+w0wa']
#info.legendnames = ['LCDM','LCDM+Neff','LCDM+Neff+w0wa','LCDM+Neff+mnu+w0wa']
info.ticknumber = 3
info.ticksize = 14
info.line_width = 2
info.fontsize = 35
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
#info.cm = [
#	(0.,1.,0.,1.),
#        (0.,	  0.,	   0.,      1.),
#        (0.99843, 0.25392, 0.14765, 1.),
#        (0, 0, 1., 1.),
#        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
#info.cmaps = [plt.cm.Greens,plt.cm.gray_r, plt.cm.Reds_r, plt.cm.Blues,plt.cm.Greens]
info.alphas = [1.0, 0.8, 0.6, 0.4]

# use this to control the boundaries of 1d and 2d plots
#info.force_limits = {'H_0':[65:80],'tau_reio':[0.04:0.085],r'$\sum m_\nu$':[0.258,1.05]}
# use this to customise the ticks.
#info.ticknumber = 5
#info.ticksize = 10

info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']
