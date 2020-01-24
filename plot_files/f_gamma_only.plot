import matplotlib.pyplot as plt
info.redefine = {'Gamma_dcdm':'(1.02e-3)*Gamma_dcdm'}
info.to_change = {'f_dcdm':r'$f_{\mathrm{dcdm}}$','Gamma_dcdm':r'$\Gamma_{\mathrm{dcdm}}$ [Gyr$^{-1}$]'}
#info.to_plot = [r'$\Gamma_{\mathrm{dcdm}}$',r'$f_{\mathrm{dcdm}}$','H0','sigma8','omega_totcdm']
#info.to_plot = [r'$\Gamma_{\mathrm{dcdm}}$',r'$f_{\mathrm{dcdm}}$','H0','sigma8','omega_totcdm']
#info.to_plot = [r'$f_{\mathrm{dcdm}}$',r'$\Gamma_{\mathrm{dcdm}}$']
info.to_plot = [r'$f_{\mathrm{dcdm}}$',r'$\Gamma_{\mathrm{dcdm}}$ [Gyr$^{-1}$]']

#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.to_plot = ['f_dcdm','Gamma_dcdm']
#info.legendnames = ['high-$\ell$ TT','high-$\ell$ TT+TE+EE']
info.legendnames = ['CMB','CMB+BAO+WiggleZ']
#info.legendnames = ['CMB','All Experiments']
#info.legendnames = ['high-$\ell$ TT + low-$\ell$ + lensing','high-$\ell$ TT+TE+EE + low-$\ell$ + lensing']
info.ticknumber = 7
#info.new_scales = {'$\Gamma_{\mathrm{dcdm}}$': 100}
info.ticksize = 8
info.decimal = 2
info.line_width = 2
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=20
#info.cm = [(0.1, 0.1, 0.14765, 0.1),] 
#info.cm = [(0.99843, 0.25392, 0.14765, 1.),] 
#info.cm = [(0.1,0.1,1.,0.8),]
#info.cm = [(0.30235, 0.15039, 0.74804, 1.),(0.99843, 0.25392, 0.14765, 1.)]
#info.cmaps = [plt.cm.Purples,plt.cm.Reds]

#info.axis = [([0.,2000.,0.,1.])]
info.cm = [
        #(0.30235, 0.15039, 0.74804, 1.),
	(0.05,0.05,1.,1.),
        (0.99843, 0.25392, 0.14765, 1.),]
#        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
info.cmaps = [plt.cm.Blues, plt.cm.Reds]
#info.cmaps = [plt.cm.Purples, plt.cm.Reds_r, plt.cm.Greens]
info.alphas = [1.0, 0.6]


