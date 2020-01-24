import matplotlib.pyplot as plt
#info.redefine={'log10PBH_high_mass':'10**log10PBH_high_mass'}

#info.redefine={'log10_Omega_fld':'10**log10_Omega_fld'}
info.redefine={'Omega_many_fld':'Omega_many_fld*H0*H0/100/100','log10_a_c':'10**log10_a_c','Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(1e-8)**(-3)+(2.47310e-5*1.445)*(1e-8)**(-4)+Omega_Lambda)'}

#info.redefine={'log10PBH_low_mass':'10**log10PBH_low_mass','log10PBH_fraction':'10**log10PBH_fraction'}
#info.to_change={'100*theta_s':r'$100~\theta_s$','PBH_high_mass':r'$M_{\rm PBH}/M_\odot$','PBH_fraction':r'$f_{\rm PBH}$','log10PBH_high_mass':r'$M_{\rm PBH}/M_\odot$','log10PBH_low_mass':r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$','log10PBH_fraction':r'${\rm Log}_{10}f_{\rm PBH}$'}
info.to_change={'Omega_many_fld':r'$\omega_{\rm ALP}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$'}
#info.legendnames = ['litecore80','litecore120','COrE+']
#info.legendnames = ['Planck no Lensing','Planck Full','Planck+BAO+fsig8']
#info.legendnames = ['All Data','Planck Full','Planck no Lensing']
#info.legendnames = ['Ricotti','AliHaimoud','Gaggero','Horowitz']
#info.legendnames = ['instanteneous reionization','asymmetric reionization']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 8
info.ticksize = 8
info.decimals=2
info.fontsize = 20
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','100*theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$\omega_{\rm ALP}$','log10PBH_fraction','PBH_fraction','log10_Omega_fld','H0','Omega_Lambda','Omega_fld','Omega_many_fld','Omega_axion_ac','Theta_initial_fld']

info.to_plot = ['omega_cdm',r'$\omega_{\rm ALP}$','a_c','Omega_Lambda',r'$f_{\rm axion}(a_c)$']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$f_{\rm PBH}$','log10PBH_low_mass','log10PBH_fraction',r'$M_{\rm PBH}[g]$']
#info.to_plot = [r'${\rm Log}_{10}f_{\rm PBH}$',r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=20
