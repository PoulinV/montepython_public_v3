import matplotlib.pyplot as plt
#info.redefine = {'Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'Gamma_dcdm':'f_dcdm Gamma_dcdm' }
info.legendnames = ['core+euclid / kmax = 0.2','core+desi+euclid+21cm / kmax = 0.2','core+euclid / kmax = 0.1','core+desi+euclid+21cm / kmax = 0.1']
#info.legendnames = ['core','core+desi','core+desi+euclid','core+desi+euclid+21cm']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 4
info.ticksize = 15
info.fontsize = 25
info.line_width = 2
info.to_change = {'M_tot':r'$\sum m_\nu$'}
info.to_plot = [r'$\sum m_\nu$','tau_reio','omega_cdm','H0', 'ln10^{10}A_s','n_s']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
#info.bins=5
