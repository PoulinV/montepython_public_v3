import matplotlib.pyplot as plt
#info.redefine = {'Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'Gamma_dcdm':'f_dcdm Gamma_dcdm' }
info.redefine = {'omega_totcdm':'(omega_totcdm+omega_b/100)*10000/(H0**2)','Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'omega_totcdm':'omega_cdm'}
info.to_change = {'omega_totcdm':r'$\Omega_M$','Gamma_dcdm':r'$f_{dcdm} \Gamma_{dcdm}$'}
#info.to_plot = ['Omega_m','sigma8','H0',r'$f_{dcdm} \Gamma_{dcdm}$']
info.to_plot = [r'$\Omega_M$','sigma8','H0',r'$f_{dcdm} \Gamma_{dcdm}$']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 5
info.ticksize = 12
info.legendnames=['CMB','HST+WL+BAO+Planck Cluster','All Experiments']
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
#info.bins=5
info.line_width = 2

