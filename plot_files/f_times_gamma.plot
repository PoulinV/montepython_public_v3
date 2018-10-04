import matplotlib.pyplot as plt
#info.redefine = {'Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'f_dcdm':'$f_{\mathrm{dcdm}}$','Gamma_dcdm':r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$'}
#info.to_change = {'omega_totcdm':'omega_cdm'}
#info.to_change = {'Gamma_dcdm':r'$\Gamma_{\mathrm{dcdm}}$'}
info.to_change = {'f_dcdm':r'$f_{\mathrm{dcdm}}$','omega_totcdm':'omega_cdm','Gamma_dcdm':r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$ [Gyr$^{-1}$]'}
#info.to_change = {'Gamma_dcdm':r'$\Gamma_{\mathrm{dcdm}}$','omega_totcdm':'omega_cdm','f_dcdm':r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$'}
info.to_plot = [r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$ [Gyr$^{-1}$]']
#info.to_plot = [r'$f_{\mathrm{dcdm}}$',r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$','omega_cdm','sigma8','H0']
#info.to_plot = [r'$\Gamma_{\mathrm{dcdm}}$',r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$','omega_cdm','sigma8','H0']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 5
info.ticksize = 8
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
#info.bins=15
#info.x_range = [0.1,14]
info.legendnames = ['high-$\ell$ TT','high-$\ell$ TT+TE+EE']
#info.legendnames = ['CMB','HST+WL+BAO+Planck Clusters','All Experiments']
#info.legendnames = ['CMB','CMB+BAO+WiggleZ']
info.legendsize = 10
#info.legendnames = ['','']
#info.cm = [(0.99843, 0.25392, 0.14765, 1.),] 
info.cmaps = [plt.cm.Blues,plt.cm.Reds]
info.cm = [
	#(0.30235, 0.15039, 0.74804, 1.),
        (0.05,0.05,1.,1.),
        (0.99843, 0.25392, 0.14765, 1.),]
#        (0.90000, 0.75353, 0.10941, 1.)]
#info.cm = [(0.30235, 0.15039, 0.74804, 1.),]
info.line_width = 2
info.alphas = [1.0, 0.6]

