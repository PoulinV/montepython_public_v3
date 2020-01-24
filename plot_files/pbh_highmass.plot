import matplotlib.pyplot as plt
#info.redefine = {'Gamma_dcdm': '(f_dcdm*Gamma_dcdm)'}
#info.to_change = {'Gamma_dcdm':'f_dcdm Gamma_dcdm' }
info.to_change={'100*theta_s':r'$100~\theta_s$','PBH_mass':r'$M_{\rm PBH}$','PBH_fraction':r'$f_{\rm PBH}$'}
#info.legendnames = ['litecore80','litecore120','COrE+']
info.legendnames = ['Ricotti','AliHaimoud','Gaggero','Horowitz']
#info.legendnames = ['instanteneous reionization','asymmetric reionization']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 3
info.ticksize = 15
info.decimals=3
info.fontsize = 25
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}$',r'$f_{\rm PBH}$']
info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}$',r'$f_{\rm PBH}$']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=15