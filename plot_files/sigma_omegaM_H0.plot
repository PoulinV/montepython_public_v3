import matplotlib.pyplot as plt
#info.redefine = {'Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'Gamma_dcdm':'f_dcdm Gamma_dcdm' }
#info.redefine = {'omega_cdm':'(omega_cdm+0.02225)*10000/(H0*H0)'}
info.redefine = {'omega_totcdm':'(omega_totcdm+0.0225)*10000/(H0*H0)'}
#info.redefine = {'omega_totcdm':'(omega_totcdm+0.0225)*10000/(H0*H0)','Gamma_dcdm':'(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'omega_m':'Omega_m'}
#info.to_change = {'omega_totcdm':'Omega_m','Gamma_dcdm':r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$'}
info.to_change = {'omega_totcdm':'Omega_m','Gamma_dcdm':r'$\Gamma_{\mathrm{dcdm}}$','f_dcdm':r'$f_{\mathrm{dcdm}}$'}
#info.to_plot = ['Omega_m','sigma8','H0',r'$f_{\mathrm{dcdm}} \Gamma_{\mathrm{dcdm}}$']
info.to_plot = ['Omega_m','sigma8','H0',r'$f_{\mathrm{dcdm}}$']
info.legendnames = ['CMB','HST+WL+BAO+Planck Clusters','All Experiments']
#info.legendnames = ['CMB','without z=2.36 BAO data','with z=2.36 BAO data','hst Riess 2016']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 5
info.ticksize = 12
info.decimals = 2
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=10
info.line_width = 2
info.legendsize = 20
