'''
Your midterm is to run an MCMC fit for an RV planet's orbit. This should be a Metropolis-Hastings fitter with a Gibbs sampler, which is what we talked about in class. No, you are not allowed to just send the function to emcee and let a miracle occur. You should code up the fitting routine using your language of choice.

To simplify matters, we're going to use a Hot Jupiter with a circular orbit, and we aren't going to try fitting transit data or anything silly like that. This means that while you can use your orbit predictor code if you want, it's probably just easier to do this analytically using a sine curve. To be specific...

---You should plan to fit three parameters: orbital period, Msin(i), and a time zero point. (Note that for a circular orbit, longitude of periapse is formally undefined, so I suggest defining the time zero point to be when the planet is crossing in front of the star with zero radial velocity.)

---Don't worry about multiple walkers or anything like that. Do worry about making sure you discard early steps before burn-in.

---You should track the acceptance rate on proposed jumps for each parameter. Recall from our discussion in class that you don't want a value that's too high (or it takes a long time to mix) or too low (wasting a lot of computation power on failed jumps). Aim for 20 to 40 percent, and adjust your jump sizes accordingly.

---You should report the median value for each parameter, as well as the range encompassing the central 68% (i.e., a 1ish sigma confidence interval).

---You also should produce contour plots of the 2D posterior for all combinations of parameters: Msin(i) vs period, Msin(i) vs time zero point, and period versus time zero point. Have the contours enclose useful amounts of the posterior (i.e., 50/90//99/99.9 percent, or 68/95/99.7 percent).

---You should produce a phased RV curve, showing the data points (with error bars) and your best-fit solution. Cautionary note: Pay attention to what's going on around phase of zero, and decide if that specific data should be constraining your model.

---In addition to the things in the previous 4 bullet points, which should be reported in a concise summary, you should upload your code to github.

The system that we'll be using is HD 209458. Luckily Stefano has a set of RVs from Keck online that we can adopt:

https://github.com/stefano-meschiari/Systemic2/blob/master/datafiles/HD209458_3_KECK.vels

Feel free to go look up the actual system parameters, or pretty much anything else you think you need. No discussion amongst yourselves though. Let's make it due on November 5.

'''

import numpy as np
import scipy.constants as const
from scipy.stats import norm
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
from matplotlib import font_manager
fontproperties = {'family':'sans-serif','sans-serif':['cmss'],'weight':'normal','size':12}
ticks_font = font_manager.FontProperties(family='cmss', style='normal', size=9, weight='normal', stretch='normal')
from matplotlib import rc
rc('text', usetex=True)
rc('font', **fontproperties)




def getdata():
   
   HJD, RV, unc = [],[],[]
   data = open('HD209458_3_KECK.vels','r')
   for line in data.readlines():
      if not line.startswith('#'):
         thisline = line.split()
         HJD.append(float(thisline[0]))
         RV.append(1e2*float(thisline[1]))
         unc.append(1e2*float(thisline[2]))
         
   data.close()

   ### Table 1 in http://adsabs.harvard.edu/abs/2005ApJ...629L.121L
   return np.array(HJD), np.array(RV), np.array(unc)        ### RV and uncertainty converted to cm s-1


def RVeqn(t,param):
   P, m2sinI, t0 = param[0],param[1],param[2]
   m1 = 1.14 *M_Sun
   return (2 *pi() *G /m1**2)**(1/3.) /(P**(1/3.)) *m2sinI *cos(2 *pi() /P *day_to_sec(t-t0))
   #a = 0.04747 *AU
   #return 2 *pi() /P /m1 *a *m2sinI *cos(2 *pi() /P *(t-t0))


   
def init_P():
   return np.random.rand() *(day_to_sec(4) -day_to_sec(3)) +day_to_sec(3)     ### uniform (3, 4) days

def prop_P(P):
   new_P = 0
   while not (3 < sec_to_day(new_P) < 4):
      new_P = (np.random.rand()-0.5) *0.5*day_to_sec(1) +P
   return new_P


def init_m2sinI():
   return np.random.rand() *(1*M_Jup -0.4*M_Jup) +0.4*M_Jup           ### uniform (0.4 M_Jup, 1 M_Jup)


def prop_m2sinI(m2sinI):
   new_m2sinI = 0
   while not (0.1 < new_m2sinI/M_Jup < 2):
      new_m2sinI = (np.random.rand()-0.5) *0.5*M_Jup +m2sinI
   return new_m2sinI


def init_t0():
   return np.random.rand() *(2453371 -2451341) +2451341                   ### uniform (HJD 2451341, 2453371)


def prop_t0(t0):
   new_t0 = 0
   while not (2451341 < new_t0 < 2453371):
      new_t0 = (np.random.rand()-0.5) *2e3 +t0
   return new_t0


def fitdata(steps):
   ### fit three parameters: orbital period, Msin(i), and a time zero point
   
   HJD, RV, uncertainty = getdata()
   data = [HJD, RV, uncertainty]
   
   P, m2sinI, t0 = init_P(), init_m2sinI(), init_t0()
   param = [P, m2sinI, t0]
   chi2 = chi2_vsdata(data,param)

   ### track acceptance rates
   acc_P, acc_m2sinI, acc_t0 = [],[],[]
   
   ### track MC through parameter space
   sav_P, sav_m2sinI, sav_t0, sav_chi2 = [],[],[],[]
   
   counter, jump_ct = 0,0
   while counter < steps:
      
      if counter%3==0:
         new_P = prop_P(param[0])
         prop_param = [new_P, param[1], param[2]]
         param_stdev = day_to_sec(1)
      
      elif counter%3==1:
         new_m2sinI = prop_m2sinI(param[1])
         prop_param = [param[0], new_m2sinI, param[2]]
         param_stdev = 0.2*M_Jup

      elif counter%3==2:
         new_t0 = prop_t0(param[2])
         prop_param = [param[0], param[1], new_t0]
         param_stdev = 20
      
      prob_acc = min(1., exp(0.5 *(chi2_vsdata(data,param) -chi2_vsdata(data,prop_param))) *prop_dist_given_prop(param[counter%3],prop_param[counter%3],param_stdev) /prop_dist_given_prior(prop_param[counter%3],param[counter%3],param_stdev) ) #*prop_dist_given_prior(prop_param[counter%3],param[counter%3],param_stdev)
         
      if counter%3==0: acc_P.append(prob_acc)
      elif counter%3==1: acc_m2sinI.append(prob_acc)
      elif counter%3==2: acc_t0.append(prob_acc)

      if np.random.rand() < prob_acc:
         param, chi2 = prop_param, chi2_vsdata(data,prop_param)
         jump_ct += 1
      
      if jump_ct>=1e3 and counter%100==0:
         sav_P.append(param[0])
         sav_m2sinI.append(param[1])
         sav_t0.append(param[2])
         sav_chi2.append(chi2)

      if counter%10==0:
         print(jump_ct, counter, '%10s'%'%.5e'%prob_acc, '%10s'%'%.5g'%posterior(data,param), '%10s'%'%.5g'%posterior(data,prop_param), '%10s'%'%.1f'%chi2, '%10s'%'%.5g'%sec_to_day(param[0]) +' days', '%10s'%'%.3g'%(param[1]/M_Jup) +' M_Jup', '%14s'%'HJD %.2f'%param[2], '%10s'%'%.1f'%chi2_vsdata(data,prop_param), '%10s'%'%.5g'%sec_to_day(prop_param[0]) +' days', '%10s'%'%.3g'%(prop_param[1]/M_Jup) +' M_Jup', '%14s'%'HJD %.2f'%prop_param[2])

      counter += 1
   
   print('')
   print('acceptance rates')
   print('  P : %.5e'%(np.mean(acc_P)))
   print('  m2sinI : %.5e'%(np.mean(acc_m2sinI)))
   print('  t0 : %.5e'%(np.mean(acc_t0)))
   print('')
   print('length')
   print('  period : '+str(len(sav_P)))
   print('  m2sinI : '+str(len(sav_m2sinI)))
   print('  t0 : '+str(len(sav_t0)))
   print('')
   print('medians')
   print('  period : %.5g'%sec_to_day(np.median(sav_P)) +' days')
   print('  m2sinI : %.3g'%(np.median(sav_m2sinI)/M_Jup) +' M_Jup')
   print('  t0 : HJD %.2f'%np.median(sav_t0))
   print('  chi2 : %.1f'%np.median(sav_chi2))
   print('')
   print('means')
   print('  period : %.5g'%sec_to_day(np.mean(sav_P)) +' days')
   print('  m2sinI : %.3g'%(np.mean(sav_m2sinI)/M_Jup) +' M_Jup')
   print('  t0 : HJD %.2f'%np.mean(sav_t0))
   print('  chi2 : %.1f'%np.mean(sav_chi2))
   print('')
   print('standard deviations')
   print('  period : %.5g'%sec_to_day(np.std(sav_P)) +' days')
   print('  m2sinI : %.3g'%(np.std(sav_m2sinI)/M_Jup) +' M_Jup')
   print('  t0 : HJD %.2f'%np.std(sav_t0))
   print('  chi2 : %.1f'%np.std(sav_chi2))
   print('')
   print('minima')
   print('  period : %.5g'%sec_to_day(np.min(sav_P)) +' days')
   print('  m2sinI : %.3g'%(np.min(sav_m2sinI)/M_Jup) +' M_Jup')
   print('  t0 : HJD %.2f'%np.min(sav_t0))
   print('  chi2 : %.1f'%np.min(sav_chi2))
   print('')
   print('maxima')
   print('  period : %.5g'%sec_to_day(np.max(sav_P)) +' days')
   print('  m2sinI : %.3g'%(np.max(sav_m2sinI)/M_Jup) +' M_Jup')
   print('  t0 : HJD %.2f'%np.max(sav_t0))
   print('  chi2 : %.1f'%np.max(sav_chi2))
   print('')

   return np.array(sav_P), np.array(sav_m2sinI), np.array(sav_t0), np.array(sav_chi2)



def prop_dist_given_prop(prior_param,prop_param,stdev):
   return norm.pdf(prior_param,prop_param,stdev)


def prop_dist_given_prior(prop_param,prior_param,stdev):
   return norm.pdf(prop_param,prior_param,stdev)


def posterior(data,param):
   return chi2_vsdata(data,param)



def chi2_vsdata(data,param):

   HJD = data[0]
   RV_data = data[1]
   unc = data[2]
   
   RV_model = RVeqn(HJD,param)
   
   chi2 = 0.
   for i in range(len(HJD)):
      chi2 += (RV_data[i] -RV_model[i])**2 / unc[i]**2
   return chi2



def plot_fit(P, m2sinI, t0, chi2):

   data = getdata()
   plt.close()
   fig = plt.figure(1,figsize=(6,6))
   ax = fig.add_subplot(111)
   
   xfit = np.arange(np.median(t0),np.median(t0)+sec_to_day(np.median(P)),1e-2)
   RV = RVeqn(xfit,[np.median(P), np.median(m2sinI), np.median(t0)])
   
   for i in range(len(xfit)): xfit[i] = xfit[i] %sec_to_day(np.median(P))
   
   xdata = data[0]
   xdata = xdata -np.median(t0)
   xdata = xdata %sec_to_day(np.median(P))
   print(len(xdata))
   
   ax.scatter(xfit,RV,color='green',marker='o',s=0.5,lw=1,alpha=1,label='MCMC fit')
   ax.errorbar(xdata,data[1],yerr=[data[2],data[2]],fmt='+',lw=1,c='red')
   ax.scatter(xdata,data[1],marker='o',facecolor='white',edgecolor='red',lw=0.6,s=20,label='data')
   
   
   '''
   ax.set_xlabel(r'$\lambda$ (\AA)')
   ax.set_xscale('log')
   ax.set_xlim([1e4,1e14])
   ax.set_ylabel(r'$f_{\nu}$ (Jy)')
   ax.set_yscale('log')
   ax.set_ylim([1e-42,1e-26])

   legend = ax[j].legend(loc='upper right',shadow=False,labelspacing=-0.01,borderpad=0.2,title=case+' at '+str(D[j])+' AU',frameon=False)
   for label in legend.get_texts(): label.set_fontsize('small')
   for label in legend.get_lines(): label.set_linewidth(2)
   
   '''

   fig.tight_layout()
   plt.savefig('plot_fit.pdf')


def plot_contour(P, m2sinI, t0, chi2):

   per = np.percentile(chi2,np.arange(1,101,1))    #[68,95,99.7])

   x, y = sec_to_day(P), m2sinI/M_Jup
   z = np.empty([len(x),len(y)])
   for i in range(len(z)):
      for j in range(len(z[i])):
         z[i,j] = np.interp(chi2[i],per,np.arange(1,101,1))
   
   '''
   plt.clf()
   CS = plt.contour(x,y,z,levels=[68,95,99.7])
   plt.clabel(CS, inline=1, fontsize=10)
   plt.xlabel('Period (d)')
   plt.ylabel('$m_2 \sin i$ ($M_{\mathrm{Jup}}$)')
   plt.savefig('plot_contour_1.pdf')
   
   '''

   plt.close()
   plt.scatter(sec_to_day(P), m2sinI/M_Jup, alpha=0.2, lw=0.2, s=2)
   plt.xlabel('Period (d)')
   plt.ylabel('$m_2 \sin i$ ($M_{\mathrm{Jup}}$)')
   plt.savefig('plot_scatter_1.pdf')

   x, y = sec_to_day(P), t0
   z = np.empty([len(x),len(y)])
   for i in range(len(z)):
      for j in range(len(z[i])):
         z[i,j] = np.interp(chi2[i],per,np.arange(1,101,1))

   '''
   plt.clf()
   CS = plt.contour(x,y,z,levels=[68,95,99.7])
   plt.clabel(CS, inline=1, fontsize=10)
   plt.xlabel('Period (d)')
   plt.ylabel('$t_{\mathrm{peri}}$ (HJD)')
   plt.savefig('plot_contour_2.pdf')

   '''

   plt.close()
   plt.scatter(sec_to_day(P), t0, alpha=0.2, lw=0.2, s=2)
   plt.xlabel('Period (d)')
   plt.ylabel('$t_{\mathrm{peri}}$ (HJD)')
   plt.savefig('plot_scatter_2.pdf')

   x, y = m2sinI/M_Jup, t0
   z = np.empty([len(x),len(y)])
   for i in range(len(z)):
      for j in range(len(z[i])):
         z[i,j] = np.interp(chi2[i],per,np.arange(1,101,1))

   '''
   plt.clf()
   CS = plt.contour(x,y,z,levels=[68,95,99.7])
   plt.clabel(CS, inline=1, fontsize=10)
   plt.xlabel('$m_2 \sin i$ ($M_{\mathrm{Jup}}$)')
   plt.ylabel('$t_{\mathrm{peri}}$ (HJD)')
   plt.savefig('plot_contour_3.pdf')

   '''

   plt.close()
   plt.scatter(m2sinI/M_Jup, t0, alpha=0.2, lw=0.2, s=2)
   plt.xlabel('$m_2 \sin i$ ($M_{\mathrm{Jup}}$)')
   plt.ylabel('$t_{\mathrm{peri}}$ (HJD)')
   plt.savefig('plot_scatter_3.pdf')



### define constants in cgs units
h     = const.h *1e7           ### erg s
c     = const.c *1e2           ### cm s-1
k     = const.k *1e7           ### erg K-1
G     = const.G *1e3           ### cm3 g-1 s-2
AU    = 1.496e13               ### cm
sigma = const.sigma *1e3       ### g s-3 K-4
R_Sun   = 6.955e10             ### cm
M_Sun   = 1.989e33             ### grams
M_Jup   = 1.898e30             ### grams
M_Earth = 5.972e27             ### grams


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
def hrs_to_sec(hrs): return hrs *60. *60.
def cm_to_AU(cm): return cm *6.68458712e-14
def AU_to_cm(AU): return AU *1.496e13
def cm_to_Rsun(cm): return cm /R_Sun
def Rsun_to_cm(Rsun): return Rsun *R_Sun
def pc_to_cm(pc): return pc *3.086e18
def g_to_Mearth(g): return g /5.972e27



if __name__ == '__main__':
   steps = input('Enter number of steps: ')
   P, m2sinI, t0, chi2 = fitdata(steps)
   plot_fit(P, m2sinI, t0, chi2)
   plot_contour(P, m2sinI, t0, chi2)

