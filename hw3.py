'''
   Andrew Leung
   UT Austin
	
   AST 381  Planetary Astrophysics
   Professor Adam Kraus
	
   Homework #3
   20 Oct 2015
	
'''


import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
from matplotlib import font_manager
fontproperties = {'family':'sans-serif','sans-serif':['cmss'],'weight':'normal','size':10}
ticks_font = font_manager.FontProperties(family='cmss', style='normal', size=8, weight='normal', stretch='normal')
from matplotlib import rc
rc('text', usetex=True)
rc('font', **fontproperties)


h     = const.h *1e7           ### erg s
c     = const.c *1e2           ### cm s-1
k     = const.k *1e7           ### erg K-1
G     = const.G *1e3           ### cm3 g-1 s-2
AU    = 1.496e13               ### cm
R_Sun = 6.955e10               ### cm
sigma = const.sigma *1e3       ### g s-3 K-4


'''
   The goal of this assignment is to build a toy model of a debris disk.
	
'''


def planck_nu(nu,T_e):
	return 2 * h * nu**3 / c**2 / (exp(h*nu/k/T_e)-1)       ### erg s-1 cm-2 Hz-1 sr-1


def planck_lambda(wl,T_e):
	return 2 * h * c**2 / wl**5 / (exp(h*c/wl/k/T_e)-1)     ### erg s-1 cm-2 cm-1 sr-1


def part_1(case,plot):

	if case == 'Fomalhaut':
		T = 8590.             ### Fomalhaut surface temperature in kelvin
		R = 1.842             ### Fomalhaut radius in solar unit
		D = [10.,130.]        ### user-specified orbital radius (in AU)
	
	elif case == 'Sun':
		T = 5800.             ### Sun surface temperature in kelvin
		R = 1.                ### Sun radius in solar unit
		D = [1.,10.]          ### user-specified orbital radius (in AU)

	lambda_list = 10**(np.linspace(-7,-1,6e3))
	f_nu_list = [[],[]]
	f_lambda_list = [[],[]]
	
	for j in range(len(D)):
		for i in range(len(lambda_list)):
			f_nu_list[j].append(planck_lambda(lambda_list[i],T) *pi()*(Rsun_to_cm(R)/AU_to_cm(D[j]))**2 *lambda_list[i]**2 /c *1e23)                   ### in Jansky
			f_lambda_list[j].append(planck_lambda(lambda_list[i],T) *pi()*(Rsun_to_cm(R)/AU_to_cm(D[j]))**2 *1e-8)                     ### in erg s^-1 cm^-2 A^-1

	if plot:
		plt.close()
		fig = plt.figure(1,figsize=(8.1,3.6))
		ax  = [fig.add_subplot(121),fig.add_subplot(122)]
		plot_list = [f_nu_list,f_lambda_list]
		
		for j in range(len(ax)):
			for i in range(len(D)):
				ax[j].plot(1e8*lambda_list,plot_list[j][i],color=['blue','green'][i],marker='',ls=['-','--'][i],ms=1,lw=1.6+0.4*i,label='at '+str(int(D[i]))+' AU')

			ax[j].set_xlabel(r'$\lambda$ (\AA)')
			ax[j].set_xscale('log')
			ax[j].set_xlim([6e2,[1e7,2e5][j]])

			if j == 0: ax[j].set_ylabel(r'$f_{\nu}$ (Jy\,=\,$10^{-23}$\,erg s$^{-1}$\,cm$^{-2}$\,Hz$^{-1}$)')
			elif j == 1: ax[j].set_ylabel(r'$f_{\lambda}$ (erg s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$)')
			ax[j].set_yscale('log')
			ax[j].set_ylim([10**int(np.log10(np.max(plot_list[j][1]))+[-2,-3][j]),10**(int(np.log10(np.max(plot_list[j][0])))+1)])

			legend = ax[j].legend(loc='upper right',shadow=False,labelspacing=0.12,borderpad=0.2,title=case,frameon=False)
			for label in legend.get_texts(): label.set_fontsize('small')
			for label in legend.get_lines(): label.set_linewidth(2)

		fig.tight_layout()
		if case == 'Fomalhaut': plt.savefig('part_1.pdf')
		elif case == 'Sun': plt.savefig('part_1_test.pdf')

	return lambda_list,f_nu_list,f_lambda_list


def parts_2_3_4(case):		
	
	lambda_list,f_nu_list,f_lambda_list = part_1(case,False)
	if case == 'Fomalhaut': D,R = [10,130],1.842
	elif case == 'Sun': D,R = [1,10],1.

	size = ['1e-1','1e+0','1e+1','1e3']      ### microns
	
	plt.close()
	fig = plt.figure(1,figsize=(8.1,3.6))
	ax  = [fig.add_subplot(121),fig.add_subplot(122)]
	dust_lambda_list = 10**(np.linspace(-4,8,6e3))
	
	for i in range(len(size)):
		
		if i < 3:
			wl_micron,Q_abs = [],[]
			data = open(size[i]+'micron.dat','r')
			for line in data.readlines():
				if not line.startswith('#'):
					thisline = line.split()
					wl_micron.append(float(thisline[0]))      ### microns
					Q_abs.append(float(thisline[1]))
			data.close()
		
			wl_cm = []
			for j in range(len(wl_micron)):
				wl_cm.append(float('%.4g'%(1e-4*wl_micron[j])))
				
			for j in range(len(f_nu_list)):
				Q_abs_interp = []
				for k in range(len(lambda_list)):
					Q_abs_interp.append(np.interp(lambda_list[k],wl_cm[::-1],Q_abs[::-1]))

				abs_power = 0
				for k in range(len(lambda_list)-1):
					abs_power += f_lambda_list[j][k] *Q_abs_interp[k] *1e8*(lambda_list[k+1]-lambda_list[k]) *pi()*(1e-4*float(size[i]))**2
				T_eq = (abs_power / (4 *sigma))**0.25      ### this is equilibrium temperature assuming blackbody emission
				print('size (microns) = '+size[i]+', distance (AU) = '+str(D[j])+', absorbed power (erg/s) = '+'%.3g'%abs_power+', equilibrium temperature (K) = '+'%.3g'%T_eq)

				f_lambda_dust = []
				for k in range(len(dust_lambda_list)):
					f_lambda_dust.append(planck_lambda(dust_lambda_list[k],T_eq) *4*pi() *np.interp(dust_lambda_list[k],wl_cm[::-1],Q_abs[::-1]) )      ### in erg s^-1 cm^-2 cm^-1
				emit_power = 0
				for k in range(len(dust_lambda_list)-1):
					emit_power += f_lambda_dust[k] *np.interp(dust_lambda_list[k],wl_cm[::-1],Q_abs[::-1]) *(dust_lambda_list[k+1]-dust_lambda_list[k]) *4*pi()*(1e-4*float(size[i]))**2
				
				while emit_power < abs_power:
					test_emit_power = 0
					test_f_lambda_dust = []
					if emit_power < 0.3*abs_power: T_eq *= 1.1
					elif emit_power < 0.7*abs_power: T_eq *= 1.01
					else: T_eq *= 1.0005
					for k in range(len(dust_lambda_list)):
						test_f_lambda_dust.append(planck_lambda(dust_lambda_list[k],T_eq) *4*pi() *np.interp(dust_lambda_list[k],wl_cm[::-1],Q_abs[::-1]) )      ### in erg s^-1 cm^-2 cm^-1
					for k in range(len(dust_lambda_list)-1):
						test_emit_power += test_f_lambda_dust[k] *np.interp(dust_lambda_list[k],wl_cm[::-1],Q_abs[::-1]) *(dust_lambda_list[k+1]-dust_lambda_list[k]) *4*pi()*(1e-4*float(size[i]))**2
					emit_power = test_emit_power

				print('size (microns) = '+size[i]+', distance (AU) = '+str(D[j])+', emitted power (erg/s) = '+'%.3g'%emit_power+', equilibrium temperature (accounting for Q_abs) = '+'%.3g'%T_eq+' K')

				f_nu_dust = np.array(test_f_lambda_dust) *pi()*(1e-4*float(size[i])/pc_to_cm(7.70))**2 *np.array(dust_lambda_list)**2 /c *1e23      ### in Jy
				ax[j].plot(1e8*dust_lambda_list,f_nu_dust,color=['blue','green','orange'][i],marker='',ls=['-','--','-.'][0],ms=1,lw=2-0.1*i,alpha=0.6+0.2*i,label=['0.1','1.0','10'][i]+r'\,$\mu$m astrosilicate')
				print('                 f_nu at peak (Jy) = %.3g'%max(f_nu_dust))
				print('             number of dust grains = %.3g'%([0.7,10.][j]/max(f_nu_dust)))
				print('mass of debris ring (Earth masses) = %.3g'%(g_to_Mearth([0.7,10.][j]/max(f_nu_dust)*(4./3*pi()*(1e-4*float(size[i]))**3)*2)))
				print('')


		else:
			
			for j in range(len(f_nu_list)):
				abs_power = 0
				for k in range(len(lambda_list)-1):
					abs_power += f_lambda_list[j][k] *1e8*(lambda_list[k+1]-lambda_list[k]) *pi()*(1e-4*float(size[i]))**2
				T_eq = (abs_power / (4 *sigma) )**0.25
				print('size (microns) = '+size[i]+', distance (AU) = '+str(D[j])+', absorbed power (erg/s) = '+'%.3g'%abs_power+', equilibrium temperature (K) = '+'%.3g'%T_eq)
					
				f_nu_dust = []
				for k in range(len(dust_lambda_list)):
					f_nu_dust.append(planck_lambda(dust_lambda_list[k],T_eq) *pi()*(1e-4*float(size[i])/pc_to_cm(7.70))**2 *dust_lambda_list[k]**2 /c *1e23)      ### in Jy
				ax[j].plot(1e8*dust_lambda_list,f_nu_dust,color='red',marker='',ls=':',ms=1,lw=2,alpha=1,label='1\,mm perfect absorber')
				print('                 f_nu at peak (Jy) = %.3g'%max(f_nu_dust))
				print('             number of dust grains = %.3g'%([0.7,10.][j]/max(f_nu_dust)))
				print('mass of debris ring (Earth masses) = %.3g'%(g_to_Mearth([0.7,10.][j]/max(f_nu_dust)*(4./3*pi()*(1e-4*float(size[i]))**3)*2)))
				print('')


	for j in range(len(f_nu_list)):
		ax[j].set_xlabel(r'$\lambda$ (\AA)')
		ax[j].set_xscale('log')
		ax[j].set_xlim([1e4,1e14])
		ax[j].set_ylabel(r'$f_{\nu}$ (Jy)')
		ax[j].set_yscale('log')
		ax[j].set_ylim([1e-42,1e-26])
		
		legend = ax[j].legend(loc='upper right',shadow=False,labelspacing=-0.01,borderpad=0.2,title=case+' at '+str(D[j])+' AU',frameon=False)
		for label in legend.get_texts(): label.set_fontsize('small')
		for label in legend.get_lines(): label.set_linewidth(2)
														
	fig.tight_layout()
	if case == 'Fomalhaut': plt.savefig('part_3.pdf')
	elif case == 'Sun': plt.savefig('part_3_test.pdf')


def PR_drag(v,power_incident):
	return v /c**2 * power_incident                                ### g cm s-2


def rad_pres(flux_incident,flux_absorbed,R):
	return (2*flux_incident -flux_absorbed) /c /(float(R)**2)      ### g cm-1 s-2


def part_5():

	lambda_list,f_nu_list,f_lambda_list = part_1('Fomalhaut',False)
	size = ['1e-1','1e+0','1e+1','1e3']      ### microns
	D,R,T,M = [10,130],1.842,8590.,3.819e33

	for i in range(len(size)):

		if i < 3:
			wl_micron,Q_abs = [],[]
			data = open(size[i]+'micron.dat','r')
			for line in data.readlines():
				if not line.startswith('#'):
					thisline = line.split()
					wl_micron.append(float(thisline[0]))      ### microns
					Q_abs.append(float(thisline[1]))
			data.close()
		
			wl_cm = []
			for j in range(len(wl_micron)):
				wl_cm.append(float('%.4g'%(1e-4*wl_micron[j])))
	
			print('For a '+size[i]+' micron astrosilicate:')
	
		else: print('For a 1 mm perfect absorber:')
	
		for j in range(len(D)):
			P = keplers_third_law(M,((4./3*pi()*(1e-4*float(size[i]))**3)*2),AU_to_cm(D[j]))
			
			incident_flux,absorbed_flux = 0,0
			for k in range(len(lambda_list)-1):
				incident_flux += f_lambda_list[j][k] *1e8*(lambda_list[k+1]-lambda_list[k])
				
				if i < 3:
					absorbed_flux += f_lambda_list[j][k] *np.interp(lambda_list[k],wl_cm[::-1],Q_abs[::-1]) *1e8*(lambda_list[k+1]-lambda_list[k])
				else:
					absorbed_flux += f_lambda_list[j][k] *1e8*(lambda_list[k+1]-lambda_list[k])

			F_PR = PR_drag(2*pi()*AU_to_cm(D[j])/P,incident_flux*(pi()*(1e-4*float(size[i]))**2))
			P_rad = rad_pres(incident_flux,absorbed_flux,D[j]-cm_to_AU(Rsun_to_cm(R)))
			
			print('  at '+str(D[j])+' AU, F_PR = %.3g'%(F_PR)+' g cm s-2, P_rad = %.3g'%(P_rad)+' g cm-1 s-2')
			print('     timescale for removal from system = %.3g'%(sec_to_yrs(sqrt( ((4./3*pi()*(1e-4*float(size[i]))**3)*2)*AU_to_cm(D[j]) / np.abs(F_PR -P_rad*(pi()*(1e-4*float(size[i]))**2)) )))+' years')
		print('')


def keplers_third_law(m1,m2,a):
	return ((4 *pi()**2) /(G *(m1 +m2)) *a**3 )**0.5      ### in seconds



### helper functions
def pi(): return np.pi
def ln(x): return np.log(x)
def exp(x): return np.exp(x)
def sqrt(x): return np.sqrt(x)
def sin(x): return np.sin(x)
def cos(x): return np.cos(x)
def tan(x): return np.tan(x)
def arctan(x): return np.arctan(x)
def arctan2(x,y): return np.arctan2(x,y)


### unit conversions
def rad_to_deg(rad): return rad *180. /pi()
def deg_to_rad(deg): return deg /180. *pi()
def sec_to_yrs(sec): return sec /60. /60 /24 /365.256362
def yrs_to_sec(yrs): return yrs *60. *60 *24 *365.256362
def sec_to_day(sec): return sec /60. /60 /24
def day_to_sec(day): return day *24. *60 *60
def cm_to_AU(cm): return cm *6.68458712e-14
def AU_to_cm(AU): return AU *1.496e13
def cm_to_Rsun(cm): return cm /R_Sun
def Rsun_to_cm(Rsun): return Rsun *R_Sun
def pc_to_cm(pc): return pc *3.086e18
def g_to_Mearth(g): return g /5.972e27



### if run from command line
if __name__ == '__main__':
	#part_1('Sun',True)
	#part_1('Fomalhaut',True)
	#parts_2_and_3('Sun')
	#parts_2_3_4('Fomalhaut')
	part_5()


