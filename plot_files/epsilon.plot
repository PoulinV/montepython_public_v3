import matplotlib.pyplot as plt
#info.redefine={'Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','da_rec':'(100*rd_rec)/(da_rec)','R_ds':'theta_s/theta_d'}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','log10_Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$S_8$','N_ur':r'$\Delta N_{\rm eff}$','log10_a_c':r'${\rm Log}_{10}a_c$','100*theta_s':r'$100~\theta_s$','100*theta_d':r'$100~\theta_d$'}
info.to_change={'log_epsilon_dmeff':r'${\rm Log10}\epsilon$','log_ratio_dmeff2cdm':r'${\rm Log10}f_{\chi}$'}
#info.new_scales={'theta_s':0.01,'R_ds':0.01}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$\sigma_8$','100*theta_s':r'$100~\theta_s$','N_ur':r'$\Delta N_{\rm eff}$','log10_a_c':r'${\rm Log}_{10}a_c$','Omega_Lambda':'Omega_m','rd_rec':'R_ds','100*theta_d':'theta_d','100*theta_s':'theta_s'}
#info.legendnames = ['annihilation only','all effects']
info.legendnames = ['annihilation only','scattering only','all effects']
#info.legendnames = [r'$\Lambda$CDM',r'ULA, $n=2$',r'ULA, $n=\infty$']
#info.legendnames = [r'$\Lambda$CDM',r'$n=2$',r'$n=3$',r'$n=\infty$']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 5
info.ticksize = 12
info.decimal=4
info.fontsize = 20
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
info.to_plot = [r'${\rm Log10}\epsilon$',r'${\rm Log10}f_{\chi}$']
#info.custom2d = ['add_h_contour.py']
#info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$f_{\rm PBH}$','log10PBH_low_mass','log10PBH_fraction',r'$M_{\rm PBH}[g]$']
#info.to_plot = [r'${\rm Log}_{10}f_{\rm PBH}$',r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=15
