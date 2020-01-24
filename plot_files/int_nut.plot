import matplotlib.pyplot as plt
import numpy as np
#info.redefine={'log10PBH_high_mass':'10**log10PBH_high_mass'}

#info.redefine={'log10_Omega_axion_ac':'10**log10_Omega_axion_ac/(10**log10_Omega_axion_ac+Omega_m*(10**log10_axion_ac)**(-3)+(2.47310e-5*1.445)*(10**log10_axion_ac)**(-4))','da_rec':'(100*rd_rec)/(da_rec)','R_ds':'theta_s/theta_d','Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+(omega_cdm+omega_b/100)/H0/H0*100*100*(10**log10_axion_ac)**(-3)+(2.47310e-5*1.68)/H0/H0*100*100*(10**log10_axion_ac)**(-4))','Omega_many_fld':'(Omega_many_fld*1e-13*((10**log10_axion_ac)**(-6)+1)/2)/(Omega_many_fld*1e-13*((10**log10_axion_ac)**(-6)+1)/2+(omega_cdm+omega_b/100)/H0/H0*100*100*(10**log10_axion_ac)**(-3)+(2.44e-5)*(10**log10_axion_ac)**(-4))','sigma8':'sigma8*(Omega_m/0.3)**0.5','f_axion':'np.log10(f_axion)','m_axion':'np.log10(m_axion)'}
info.redefine={'sigma8':'sigma8*(Omega_m/0.3)**0.5'}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','log10_z_c':r'${\rm log}_{10}(z_c)$','log10_Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$S_8$','N_ur':r'$\Delta N_{\rm eff}$','log10_axion_ac':r'${\rm Log}_{10}a_c$','100*theta_s':r'$100~\theta_s$','100*theta_d':r'$100~\theta_d$','f_axion':r'$\alpha$','m_axion':r'$\mu$','scf_parameters__1':r'$\theta_i$','f_axion_ac':r'$f_{\rm EDE}(a_c)$','fraction_axion_ac':r'$f_{\rm EDE}(a_{\rm eq})$','cs2_fld':r'$c^2_{s,{\rm fld}}=w_{\rm fld}$'}
#info.new_scales={'theta_s':0.01,'R_ds':0.01}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','log10_axion_ac':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$\sigma_8$','100*theta_s':r'$100~\theta_s$','N_ur':r'$\Delta N_{\rm eff}$','log10_axion_ac':r'${\rm Log}_{10}a_c$','Omega_Lambda':'Omega_m','rd_rec':'R_ds','100*theta_d':'theta_d','100*theta_s':'theta_s'}
#info.legendnames = ['litecore80','litecore120','COrE+']
#info.legendnames = ['Planck no Lensing','Planck Full','Planck+BAO+fsig8']
#info.legendnames = ['All Data','Planck Full','Planck no Lensing']
#info.legendnames = ['fake $\Lambda$CDM','fake EDE','real $\Lambda$CDM','real EDE']
#info.legendnames = ['fake EDE','real EDE']
#info.legendnames = ['Planck','cmb-s4']
#info.legendnames = ['Planck+BAO+Pantheon','+SH0ES']
#info.legendnames = ['Ricotti','AliHaimoud','Gaggero','Horowitz']
#info.legendnames = ['instanteneous reionization','asymmetric reionization']
#info.legendnames = [r'$n=2$',r'$n=3$',r'$n=\infty$']
#info.legendnames = [r'$\theta_i$ free',r'$\theta_i=0.1$']
#info.legendnames = [r'$\theta_i$ free',r'$\theta_i=3$']
info.legendnames = [r'Planck 2015',r'Planck 2018']
#info.legendnames = ['with switch','without switch']
#info.legendnames = ['w/o Alens','w/ Alens']
#info.legendnames = ['TT','TTTEEE']
#info.legendnames = ['scalar field','fluid']
#info.legendnames = [r'$\Lambda$CDM',r'$n=2$',r'$n=3$',r'$n=\infty$']
#info.legendnames = [r'$\Lambda$CDM',r'ULA, $n=2$',r'ULA, $n=\infty$']
#info.legendnames = [r'$\Lambda$CDM',r'$n=2$',r'$n=3$',r'$n=\infty$']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 3
info.ticksize = 12
info.decimal=3
info.fontsize = 20
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','100*theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$\omega_{\rm ALP}$','log10PBH_fraction','PBH_fraction','log10_Omega_fld','H0','Omega_Lambda','Omega_fld','Omega_many_fld','Omega_axion_ac','Theta_initial_fld']

#info.to_plot = ['omega_cdm',r'$\omega_{\rm ALP}$','a_c','Omega_Lambda',r'$f_{\rm axion}(a_c)$','H0']
#info.to_plot = ['H_0',r'$f_{\rm axion}(a_c)$',r'${\rm Log}_{10}a_c$',r'$\mu$',r'$\alpha$',r'$\theta_{\rm ini}$','scf_parameters__1','Omega0_axion',r'$\theta_i$']
#info.to_plot = ['H_0','f_ede','scf_parameters__1','log10_z_c',r'$\theta_i$','omega_cdm',r'${\rm Log}_{10}a_c$','f_axion_ac']

#info.to_plot = ['H_0',r'$\theta_i$','r_s',r'${\rm Log}_{10}a_c$',r'$f_{\rm EDE}(a_c)$',r'$S_8$','omega_cdm','n_s','A_s','tau_reio','omega_b','n_axion','rs_rec',r'$100~\theta_s$','A_lens','A_lens_TTTEEE',r'$\Delta N_{\rm eff}$']
#info.to_plot = ['H_0',r'${\rm Log}_{10}a_c$',r'$f_{\rm EDE}(a_c)$','A_lens',r'$c^2_{s,{\rm fld}}=w_{\rm fld}$',r'$f_{\rm EDE}(a_{\rm eq})$']
info.to_plot = ['H0','log10_Geff_neutrinos','N_ur','m_ncdm']
#info.to_plot = [r'${\rm Log}_{10}a_c$',r'$f_{\rm EDE}(a_c)$','Omega_m','H_0']

#info.to_plot = ['H_0','f_ede','scf_parameters__1','log10_z_c',r'$\theta_i$','omega_cdm','log10_f_axion','log10_m_axion']
#info.to_plot = ['H_0','f_axion','log10_m_axion','f_ede','h','scf_parameters__1','log10_fraction_axion_ac','log10_axion_ac','log10_z_c','omega_cdm']
#info.to_plot = ['100*theta_s','H_0',r'$100~\theta_s$',r'$f_{\rm axion}(a_c)$',r'$\Delta N_{\rm eff}$',r'${\rm Log}_{10}a_c$',r'$\Omega_m$','tau_reio','n_s','ln10^{10}A_s','omega_cdm','omega_b','Omega_m']
#info.to_plot = [r'${\rm Log}_{10}a_c$','H_0',r'$f_{\rm axion}(a_c)$',r'$100~\theta_s$',r'$100~\theta_d$','rd_rec','da_rec','n_s','omega_cdm','omega_b','tau_reio','ln10^{10}A_s','n_s']
#info.to_plot = [r'${\rm Log}_{10}a_c$','H_0',r'$f_{\rm axion}(a_c)$',r'$\Delta N_{\rm eff}$']
#info.to_plot = [r'${\rm Log}_{10}a_c$','Omega_m','H_0',r'$S_8$',r'$f_{\rm axion}(a_c)$','R_ds','theta_s',r'$100~\theta_s$',r'$100~\theta_d$',r'$\Omega_{\rm EDE}$','n_s','ln10^{10}A_s','tau_reio']
#info.to_plot = [r'$\sigma_8$','Omega_m']
#info.custom2d = ['add_h_contour.py']
info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$f_{\rm PBH}$','log10PBH_low_mass','log10PBH_fraction',r'$M_{\rm PBH}[g]$']
#info.to_plot = [r'${\rm Log}_{10}f_{\rm PBH}$',r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$']
#info.to_plot = ['omega_b','omega_cdm', 'f_dcdm', 'log10Gamma_dcdm','M_tot' ,'100*theta_s', 'ln10^{10}A_s','n_s','tau_reio','sigma8']
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
#info.ticknumber = 10
#info.ticksize = 5
#info.to_plot = ['f_dcdm', 'Gamma_dcdm']
info.bins=16
#info.force_limits = {'log10_z_c':[3,4.5]}
info.force_limits = {r'$f_{\rm EDE}(a_c)$':[0.001,0.2],r'$f_{\rm EDE}(a_{\rm eq})$':[0.001,0.2],'H_0':[60,80]}
