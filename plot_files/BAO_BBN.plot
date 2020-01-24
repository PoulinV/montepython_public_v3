import matplotlib.pyplot as plt
#info.redefine = {'log10decay':'10**(log10decay)'}
#info.redefine = {'decay':'1e-26*decay'}
#info.to_change = {'log10decay':'Decay'}
info.to_change = {'100*theta_s':r'$100~\theta_s$','omega_dcdmdr':r'$\omega_{\rm dcdm+dr}$','M_tot':r'$\sum m_\nu$','Delta_Neff':r'$\Delta$ N$_{\rm eff}$','H0':'H_0','w0_fld':r'$w_0$','wa_fld':r'$w_a$','N_ur':r'$N_{\rm eff}$'}
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
info.to_plot = ['Omega_m','H_0','omega_b']
info.legendnames = ['Planck','BAO DR11+BBN C17','BAO DR12+BBN C17']
#info.legendnames = ['Planck','BAO+BBN C17 empirical','BAO+BBN C17 theoretical']
info.legendsize=20
info.ticknumber = 3
info.ticksize = 14
info.line_width = 2
info.fontsize = 25
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
#info.force_limits = {'H_0':'[60:95]','Omega_m':'[0.1:0.6]'}
# use this to customise the ticks.
#info.ticknumber = 5
#info.ticksize = 10

info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']

