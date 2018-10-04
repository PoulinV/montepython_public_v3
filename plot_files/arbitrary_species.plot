import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(log10decay)'}
#info.redefine = {'decay':'1e-26*decay'}
#info.to_change = {'log10decay':'Decay'}
#info.redefine = {'N_ur':'N_ur'}
info.redefine = {'arbitrary_species_density_at_knot__1a':'arbitrary_species_density_at_knot__1a+Omega_Lambda','arbitrary_species_density_at_knot__2':'arbitrary_species_density_at_knot__2+Omega_Lambda','arbitrary_species_density_at_knot__3':'arbitrary_species_density_at_knot__3+Omega_Lambda','arbitrary_species_density_at_knot__4':'arbitrary_species_density_at_knot__4+Omega_Lambda','arbitrary_species_density_at_knot__5':'arbitrary_species_density_at_knot__5+Omega_Lambda','arbitrary_species_density_at_knot__6':'arbitrary_species_density_at_knot__6+Omega_Lambda','arbitrary_species_density_at_knot__7':'arbitrary_species_density_at_knot__7+Omega_Lambda','arbitrary_species_density_at_knot__8':'arbitrary_species_density_at_knot__8+Omega_Lambda','arbitrary_species_density_at_knot__9':'arbitrary_species_density_at_knot__9+Omega_Lambda','arbitrary_species_density_at_knot__10':'arbitrary_species_density_at_knot__10+Omega_Lambda','arbitrary_species_density_at_knot__11':'arbitrary_species_density_at_knot__11+Omega_Lambda','arbitrary_species_density_at_knot__12':'arbitrary_species_density_at_knot__12+Omega_Lambda','arbitrary_species_density_at_knot__13':'arbitrary_species_density_at_knot__13+Omega_Lambda','arbitrary_species_density_at_knot__14':'arbitrary_species_density_at_knot__14+Omega_Lambda','arbitrary_species_density_at_knot__15':'arbitrary_species_density_at_knot__15+Omega_Lambda'}
#info.redefine = {'arbitrary_species_density_at_knot__1':'arbitrary_species_density_at_knot__1+1-(Omega_m+arbitrary_species_density_at_knot__1)','arbitrary_species_density_at_knot__2':'arbitrary_species_density_at_knot__2+1-(Omega_m+arbitrary_species_density_at_knot__1)','arbitrary_species_density_at_knot__3':'arbitrary_species_density_at_knot__3+1-(Omega_m+arbitrary_species_density_at_knot__1)','arbitrary_species_density_at_knot__4':'arbitrary_species_density_at_knot__4+1-(Omega_m+arbitrary_species_density_at_knot__1)','arbitrary_species_density_at_knot__5':'arbitrary_species_density_at_knot__5+1-(Omega_m+arbitrary_species_density_at_knot__1)','arbitrary_species_density_at_knot__6':'arbitrary_species_density_at_knot__6+1-(Omega_m+arbitrary_species_density_at_knot__1)','arbitrary_species_density_at_knot__7':'arbitrary_species_density_at_knot__7+1-(Omega_m+arbitrary_species_density_at_knot__1)'}
info.to_change = {'100*theta_s':r'$100~\theta_s$','M_tot':r'$\sum m_\nu$','H0':'H_0','N_ur':r'$\Delta N_{\rm eff}$','sigma8':r'$\sigma_8$','arbitrary_species_density_at_knot__1a':r'$\rho_{\rm ExDE}(z=0)$','arbitrary_species_density_at_knot__1':r'$\rho_{\rm ExDE}(z=0)$','arbitrary_species_density_at_knot__2':r'$\rho_{\rm ExDE}(z=0.15)$','arbitrary_species_density_at_knot__3':r'$\rho_{\rm ExDE}(z=0.3)$','arbitrary_species_density_at_knot__4':r'$\rho_{\rm ExDE}(z=0.5)$','arbitrary_species_density_at_knot__5':r'$\rho_{\rm ExDE}(z=0.8)$','arbitrary_species_density_at_knot__6':r'$\rho_{\rm ExDE}(z=1)$','arbitrary_species_density_at_knot__7':r'$\rho_{\rm ExDE}(z=1.5)$','arbitrary_species_density_at_knot__8':r'$\rho_{\rm ExDE}(z=2.5)$','arbitrary_species_density_at_knot__9':r'$\rho_{\rm ExDE}(z=1100)$','arbitrary_species_density_at_knot__10':r'$\rho_{\rm ExDE}(z=10^{14})$'}

#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['H_0','Omega_m',r'$\sigma_8$',r'$\sum m_\nu$']
#info.to_plot = ['omega_b','omega_cdm',r'$100~\theta_s$', 'ln10^{10}A_s','n_s','tau_reio','Omega_fld','Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta$ N$_{\rm eff}$',r'$N_{\rm eff}$']
info.to_plot = ['Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$\rho_{\rm ExDE}(z=0)$',r'$\rho_{\rm ExDE}(z=0.15)$',r'$\rho_{\rm ExDE}(z=0.3)$',r'$\rho_{\rm ExDE}(z=0.5)$',r'$\rho_{\rm ExDE}(z=0.8)$',r'$\rho_{\rm ExDE}(z=1)$',r'$\rho_{\rm ExDE}(z=1.5)$',r'$\rho_{\rm ExDE}(z=2.5)$',r'$\rho_{\rm ExDE}(z=1100)$',r'$\rho_{\rm ExDE}(z=10^{14})$','arbitrary_species_density_at_knot__9','arbitrary_species_density_at_knot__10','arbitrary_species_density_at_knot__11','arbitrary_species_density_at_knot__12','arbitrary_species_density_at_knot__13','arbitrary_species_density_at_knot__14','Omega_k']
#info.to_plot = ['Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$',r'$w_0$',r'$w_a$',r'$\Delta$ N$_{\rm eff}$',r'$N_{\rm eff}$','A_lens']
#info.to_plot = [r'$\omega_{\rm dcdm+dr}$',r'$\Gamma_{\rm dcdm}$','H0']
#info.to_plot = ['omega_b','omega_dcdmdr', 'f_dcdm', 'Gamma_dcdm', '100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8','H0','T_cmb']
#info.to_plot = [r'$\sum m_\nu$','tau_reio']
#info.legendnames = ['Planck TT+TE+EE+lowl+lensing, recfast','Core+ recfast','Core+ Hyrec']
#info.legendnames = ['LCDM+Tcmb Hyrec','LCDM Hyrec','LCDM recfast']
#info.legendnames = ['core+simlow','core+simlow+BAO']
#info.legendnames = ['LCDM recfast','LCDM hyrec']
#info.to_plot = ['Omega_m',r'$\sigma_8$','H_0',r'$\sum m_\nu$']
info.legendsize=20
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.legendnames = [r'$\sum m_\nu=0.06$ eV',r'$\sum m_\nu$ free','LCDM+mnu+w0wa','LCDM+mnu+Nfluid+w0wa']
#info.legendnames = [r'$\Omega_{\rm ExDE}\in [-\Infty;+\Infty]$',r'$\Omega_{\rm ExDE}\in [0;+\Infty]$']

#info.legendnames = ['BAO','JLA','All Data']
#info.legendnames = [r'$A_{\rm lens} = 1$',r'$A_{\rm lens}$ free']
#info.legendnames = ['Spline','Linear']
#info.legendnames = ['CMB','BBN','+BAO+SDSS']
#info.legendnames = [r'$\lambda=0$',r'$\lambda=0.1$']
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
#info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour_DES.py','add_sigma8_Omegam_contour.py']

