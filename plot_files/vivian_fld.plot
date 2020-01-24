import matplotlib.pyplot as plt
#info.redefine={'log10PBH_high_mass':'10**log10PBH_high_mass'}

#info.redefine={'log10_Omega_fld':'10**log10_Omega_fld'}
#info.redefine={'Omega_many_fld':'Omega_many_fld*H0*H0/100/100','log10_a_c':'10**log10_a_c','Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(1)**(-3)+(2.47310e-5*1.445)*(1)**(-4)+Omega_Lambda)'}
#info.redefine={'Omega_many_fld':'Omega_many_fld*1e-2'}
#info.redefine={'log10PBH_low_mass':'10**log10PBH_low_mass','log10PBH_fraction':'10**log10PBH_fraction'}
#info.to_change={'100*theta_s':r'$100~\theta_s$','PBH_high_mass':r'$M_{\rm PBH}/M_\odot$','PBH_fraction':r'$f_{\rm PBH}$','log10PBH_high_mass':r'$M_{\rm PBH}/M_\odot$','log10PBH_low_mass':r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$','log10PBH_fraction':r'${\rm Log}_{10}f_{\rm PBH}$'}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm ALP}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','Omega_many_fld__1a':r'$\Omega_{\rm a}(a_c=10^{-4})$','Omega_many_fld__2':r'$\Omega_{\rm a}(a_c=3\times10^{-3})$','Omega_many_fld__3':r'$\Omega_{\rm a}(a_c=6\times10^{-2})$'}
info.to_change={'Omega_many_fld':r'$\Omega_{\rm ALP}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','Omega_many_fld__1a':r'$\Omega_{\rm a}(a_c=8\times10^{-5})$','Omega_many_fld__2':r'$\Omega_{\rm a}(a_c=10^{-4})$','Omega_many_fld__3':r'$\Omega_{\rm a}(a_c=3\times10^{-3})$'}
#info.legendnames = ['n=1','n=2','n=3']
info.legendnames = ['w/o Neff','w/ Neff']
#info.legendnames = [r'$a_c=1e-5$',r'$a_c=1e-6$',r'$a_c=1e-7$',r'$a_c=1e-8$']
#info.legendnames = [r'$a_c=1e-5$',r'$a_c=3e-5$',r'$a_c=7e-5$',r'$a_c=1e-6$']
#info.legendnames = ['Planck no Lensing','Planck Full','Planck+BAO+fsig8']
#info.legendnames = ['All Data','Planck Full','Planck no Lensing']
#info.legendnames = ['Ricotti','AliHaimoud','Gaggero','Horowitz']
#info.legendnames = ['instanteneous reionization','asymmetric reionization']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']

info.ticknumber = 3
info.ticksize = 8
info.decimals=2
info.fontsize = 20
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','100*theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$\omega_{\rm ALP}$','log10PBH_fraction','PBH_fraction','log10_Omega_fld','H0','Omega_Lambda','Omega_fld','Omega_many_fld','Omega_axion_ac','Theta_initial_fld']
#info.to_plot = ['Omega_Lambda',r'$\Omega_{\rm ALP}$','ln10^{10}A_s','omega_b','tau_reio','omega_cdm','n_s','N_ur',r'$\Omega_{\rm a}(a_c=10^{-4})$',r'$\Omega_{\rm a}(a_c=3\times10^{-3})$',r'$\Omega_{\rm a}(a_c=6\times10^{-2})$']
info.to_plot = ['Omega_Lambda',r'$\Omega_{\rm ALP}$','ln10^{10}A_s','omega_b','tau_reio','omega_cdm','n_s','N_ur',r'$\Omega_{\rm a}(a_c=8\times10^{-5})$',r'$\Omega_{\rm a}(a_c=10^{-4})$',r'$\Omega_{\rm a}(a_c=3\times10^{-3})$']
#info.to_plot = ['N_ur',r'$\Omega_{\rm ALP}$']
#info.to_plot = ['omega_cdm',r'$\omega_{\rm ALP}$','a_c','Omega_Lambda',r'$f_{\rm axion}(a_c)$']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$f_{\rm PBH}$','log10PBH_low_mass','log10PBH_fraction',r'$M_{\rm PBH}[g]$']
#info.to_plot = [r'${\rm Log}_{10}f_{\rm PBH}$',r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=18
