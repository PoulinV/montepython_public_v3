import matplotlib.pyplot as plt
#info.redefine={'log10PBH_high_mass':'10**log10PBH_high_mass'}

#info.redefine={'log10_Omega_fld':'10**log10_Omega_fld'}
#info.redefine={'Omega_many_fld':'Omega_many_fld*H0*H0/100/100','Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(1e-5)**(-3)+(2.47310e-5*1.445)*(1e-5)**(-4)+Omega_Lambda)'}
#info.redefine={'Omega_many_fld':'Omega_many_fld*H0*H0/100/100','log10_a_c':'10**log10_a_c','Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(1e-5)**(-3)+(2.47310e-5*1.69)*(1e-5)**(-4)+Omega_Lambda)'}
#info.redefine={'Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','Omega_Lambda':'1-Omega_Lambda','rd_rec':'(100*theta_d)/(100*theta_s)'}
#info.redefine={'Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','Omega_Lambda':'1-Omega_Lambda','rd_rec':'(100*theta_d)/(100*theta_s)'}
#info.redefine={'Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','Omega_Lambda':'(1-Omega_Lambda)*(H0/100)**(3.4)'}
#info.redefine={'Omega_many_fld':'Omega_many_fld*1e-9'}
#info.redefine={'log10PBH_low_mass':'10**log10PBH_low_mass','log10PBH_fraction':'10**log10PBH_fraction'}
#info.to_change={'100*theta_s':r'$100~\theta_s$','100*theta_d':r'$100~\theta_d$','PBH_high_mass':r'$M_{\rm PBH}/M_\odot$','PBH_fraction':r'$f_{\rm PBH}$','log10PBH_high_mass':r'$M_{\rm PBH}/M_\odot$','log10PBH_low_mass':r'${\rm Log}_{10}\big(M_{\rm PBH}/g\big)$','log10PBH_fraction':r'${\rm Log}_{10}f_{\rm PBH}$'}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$\sigma_8$','N_ur':r'$\Delta N_{\rm eff}$','log10_a_c':r'${\rm Log}_{10}a_c$'}

#info.redefine={'log10_Omega_axion_ac':'10**log10_Omega_axion_ac/(10**log10_Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','da_rec':'(100*rd_rec)/(da_rec)','R_ds':'theta_s/theta_d','Omega_many_fld':'(Omega_many_fld*1e-13*((10**log10_a_c)**(-6)+1)/2)','log10_a_c':'10**log10_a_c'}
info.redefine={'log10_Omega_axion_ac':'10**log10_Omega_axion_ac/(10**log10_Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','da_rec':'(100*rd_rec)/(da_rec)','R_ds':'theta_s/theta_d','Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+(omega_cdm+omega_b/100)/H0/H0*100*100*(10**log10_a_c)**(-3)+(2.47310e-5*1.68)/H0/H0*100*100*(10**log10_a_c)**(-4))','Omega_many_fld':'(Omega_many_fld*1e-13*((10**log10_a_c)**(-6)+1)/2)/(Omega_many_fld*1e-13*((10**log10_a_c)**(-6)+1)/2+(omega_cdm+omega_b/100)/H0/H0*100*100*(10**log10_a_c)**(-3)+(2.44e-5)*(10**log10_a_c)**(-4))','sigma8':'sigma8*(Omega_m/0.3)**0.5'}
#info.redefine={'Omega_axion_ac':'Omega_axion_ac/(Omega_axion_ac+Omega_m*(10**log10_a_c)**(-3)+(2.47310e-5*1.445)*(10**log10_a_c)**(-4))','da_rec':'(100*rd_rec)/(da_rec)','R_ds':'theta_s/theta_d'}
info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','log10_Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$S_8$','N_ur':r'$\Delta N_{\rm eff}$','log10_a_c':r'${\rm Log}_{10}a_c$','100*theta_s':r'$100~\theta_s$','100*theta_d':r'$100~\theta_d$','cs2_fld':r'$c_s^2$'}
#info.new_scales={'theta_s':0.01,'R_ds':0.01}
#info.to_change={'Omega_many_fld':r'$\Omega_{\rm EDE}$','log10_a_c':'a_c','Omega_axion_ac':r'$f_{\rm axion}(a_c)$','H0':'H_0','sigma8':r'$\sigma_8$','100*theta_s':r'$100~\theta_s$','N_ur':r'$\Delta N_{\rm eff}$','log10_a_c':r'${\rm Log}_{10}a_c$','Omega_Lambda':'Omega_m','rd_rec':'R_ds','100*theta_d':'theta_d','100*theta_s':'theta_s'}
#info.legendnames = ['litecore80','litecore120','COrE+']
#info.legendnames = ['Planck no Lensing','Planck Full','Planck+BAO+fsig8']
#info.legendnames = ['All Data','Planck Full','Planck no Lensing']
#info.legendnames = ['Ricotti','AliHaimoud','Gaggero','Horowitz']
#info.legendnames = ['instanteneous reionization','asymmetric reionization']
#info.legendnames = [r'$n=2$',r'$n=3$',r'$n=\infty$']
#info.legendnames = [r'$n=3$',r'$n=\infty$']
#info.legendnames = ['Planck Only','All Data']
info.legendnames = ['TT','TTTEEE']
#info.legendnames = ['LCDM TT','EDE TT','EDE TT+H0']
#info.legendnames = [r'$c_s^2$ free',r'$c_s^2(a,k)$']
#info.legendnames = [r'$\Lambda$CDM',r'$n=2$',r'$n=3$',r'$n=\infty$']
#info.legendnames = [r'$\Lambda$CDM',r'ULA, $n=2$',r'ULA, $n=\infty$']
#info.legendnames = [r'$\Lambda$CDM',r'$n=2$',r'$n=3$',r'$n=\infty$']
info.legendsize=25
#info.to_plot = ['omega_totcdm', 'f_dcdm', 'Gamma_dcdm']
info.ticknumber = 3
info.ticksize = 12
info.decimal=4
info.fontsize = 20
info.line_width = 2
#info.to_change = {'M_tot':r'$\sum m_\nu$'}
#info.to_plot = ['tau_reio',r'$\Delta z_{\rm reio}$',r'z_{\rm end}',r'z_{\rm beg}',r'z_{\rm reio}']
#info.to_plot = ['omega_b','tau_reio','omega_cdm','100*theta_s', 'ln10^{10}A_s','n_s',r'$100~\theta_s$',r'$M_{\rm PBH}/M_\odot$',r'$\omega_{\rm ALP}$','log10PBH_fraction','PBH_fraction','log10_Omega_fld','H0','Omega_Lambda','Omega_fld','Omega_many_fld','Omega_axion_ac','Theta_initial_fld']

#info.to_plot = ['omega_cdm',r'$\omega_{\rm ALP}$','a_c','Omega_Lambda',r'$f_{\rm axion}(a_c)$','H0']
#info.to_plot = ['H_0',r'$f_{\rm axion}(a_c)$',r'${\rm Log}_{10}a_c$','cs2_fld']
#info.to_plot = ['100*theta_s','H_0',r'$100~\theta_s$',r'$f_{\rm axion}(a_c)$',r'$\Delta N_{\rm eff}$',r'${\rm Log}_{10}a_c$',r'$\Omega_m$','tau_reio','n_s','ln10^{10}A_s','omega_cdm','omega_b','Omega_m']
info.to_plot = [r'${\rm Log}_{10}a_c$','H_0',r'$f_{\rm axion}(a_c)$',r'$100~\theta_s$',r'$100~\theta_d$','rs_rec','rd_rec','da_rec','n_s','omega_cdm','omega_b','tau_reio','A_s','n_s',r'$S_8$','ln10^{10}A_s']
#info.to_plot = [r'${\rm Log}_{10}a_c$','H_0',r'$f_{\rm axion}(a_c)$',r'$\Delta N_{\rm eff}$']
#info.to_plot = [r'${\rm Log}_{10}a_c$','omega_cdm','cs2_fld','H_0',r'$f_{\rm axion}(a_c)$',r'$100~\theta_s$',r'$100~\theta_d$','n_s']
#info.to_plot = [r'${\rm Log}_{10}a_c$','H_0',r'$f_{\rm axion}(a_c)$',r'$c_s^2$']
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
info.bins=20
