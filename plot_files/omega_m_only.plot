import matplotlib.pyplot as plt
info.to_change = {'f_dcdm':r'$f_{\mathrm{dcdm}}$','Gamma_dcdm':r'$\Gamma_{\mathrm{dcdm}}$'}
info.to_plot = [r'$\Gamma_{\mathrm{dcdm}}$',r'$f_{\mathrm{dcdm}}$']
#info.to_plot = [r'$f_{\mathrm{dcdm}}$',r'$\Gamma_{\mathrm{dcdm}}$']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.legendnames = [r'$f_{\mathrm{dcdm}}${\rm vs } $\Gamma_{\mathrm{dcdm}}$']
info.ticknumber = 5
#info.new_scales = {'$\Gamma_{\mathrm{dcdm}}$': 100}
info.ticksize = 15
info.decimals = 2
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=20
#info.cm = [(0.1, 0.1, 0.14765, 0.1),] 
#info.cm = [(0.99843, 0.25392, 0.14765, 1.),] 
info.cmaps = [plt.cm.Blues]

