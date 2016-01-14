import numpy as np
import random
import scipy.constants as const
import datetime
from jdcal import gcal2jd, jd2jcal, jcal2jd
import time
import multiprocessing as mp
import gc



def pt_aux(Tladder,nchain,swapInt,burnIn,models,param_list,param_ct,psmin,psmax,jhr,real_data,it):

   verbose = False
   output = mp.Queue()
   proc = []

   for wnum in range(nchain):
      model = models[wnum]
      T = Tladder[wnum]
      proc.append(mp.Process(target=mcmc_fit, args=(swapInt,model,wnum,burnIn,verbose,param_list,param_ct,psmin,psmax,jhr,real_data,T,output,it)))
      
   for p in proc: p.start()
   for p in proc: p.join()

   results = [output.get() for p in proc]
   results.sort()


   print('      ######  results for iteration '+str(it+1)+': ')
   print('')
   print(results)
   print('')

   last_posit = []
   for wnum in range(nchain):
      last_posit.append(results[wnum][1])

   new_posit = propose_swaps(real_data,nchain,Tladder,last_posit)
   return new_posit



def propose_swaps(real_data,nchain,temp_ladder,last_posit):
   print('      ######  propose_swaps() reached at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   models = []
   for i in range(nchain): models.append([])
   
   avail_temps = np.copy(temp_ladder)
   avail_indices = np.arange(0,8)
   swap_acc_rates = np.zeros(8)
   for k in range(nchain/2):

      swap_elements = [-1,-1]
      while swap_elements[0] == swap_elements[1]:
         swap_elements = np.random.randint(nchain-2*k,size=2)
      beta_i = 1. /avail_temps[swap_elements[0]]
      beta_j = 1. /avail_temps[swap_elements[1]]
   
      chisq_sep_i, chisq_PA_i, chisq_RV_i = prob_data_given_model(real_data,last_posit[avail_indices[swap_elements[0]]])
      chisq_sep_j, chisq_PA_j, chisq_RV_j = prob_data_given_model(real_data,last_posit[avail_indices[swap_elements[1]]])

      arg = -sum(chisq_sep_i) +sum(chisq_sep_j) -sum(chisq_PA_i) +sum(chisq_PA_j) -chisq_RV_i +chisq_RV_j

      if arg >= 0: swap_prob = 1.
      else: swap_prob = exp(arg) **(beta_j -beta_i)
      swap_prob = min(1,swap_prob)

      if np.random.rand() < swap_prob:    # accept swap
         models[avail_indices[swap_elements[0]]] = last_posit[avail_indices[swap_elements[1]]]
         models[avail_indices[swap_elements[1]]] = last_posit[avail_indices[swap_elements[0]]]
         print('      ######  proposed swap accepted ')
         print('      ######     between T = { %.0f , %.0f } '%(avail_temps[swap_elements[0]],avail_temps[swap_elements[1]]))
         print('      ######    swap acceptance probability: %.3f'%swap_prob)
         print('      ######')
      else:
         models[avail_indices[swap_elements[0]]] = last_posit[avail_indices[swap_elements[0]]]
         models[avail_indices[swap_elements[1]]] = last_posit[avail_indices[swap_elements[1]]]
         print('      ######  proposed swap rejected ')
         print('      ######     between T = { %.0f , %.0f } '%(avail_temps[swap_elements[0]],avail_temps[swap_elements[1]]))
         print('      ######    swap acceptance probability: %.3f'%swap_prob)
         print('      ######')
      
      swap_acc_rates[avail_indices[swap_elements[0]]] = swap_acc_rates[avail_indices[swap_elements[1]]] = swap_prob
      avail_temps = np.delete(avail_temps,swap_elements)
      avail_indices = np.delete(avail_indices,swap_elements)
      
   #print(models)

   ### return list of length=nchain with parameter states post-swap
   ### return swap acceptance rate for each temperature
   return models, swap_acc_rates



def pt_runs(maxit,w81):
   t00 = time.time()

   param_list, param_ct, nchain, swapInt, jhr, psmin, psmax, real_data, models, temp_ladder = init_all(w81)

   swap_acc_rate = open('swap_acceptance_rate.dat','w')
   swap_acc_rate.write('# iteration \t')
   for i in range(len(temp_ladder)):
      swap_acc_rate.write(' T = %.1f \t'%(temp_ladder[i]))
   swap_acc_rate.write('\n')

   for it in range(maxit):
      t0,tl = time.time(),time.localtime()
      print('      ######  iteration %.0f of pt_runs() began at: '%(it+1) +time.strftime("%a, %d %b %Y %H:%M:%S", tl))
      
      #if (it==0): burnIn = True
      #else: burnIn = False
      burnIn = True   # [2015-12-16] always require burn-in after swaps

      if it == 0: swapInt = 2e4
      else: swapInt = 1e4  #2e3

      models, sarates = pt_aux(temp_ladder,nchain,swapInt,burnIn,models,param_list,param_ct,psmin,psmax,jhr,real_data,it)
      swap_acc_rate.write(' %.0f \t'%(it))
      for T in range(len(sarates)):
         swap_acc_rate.write(' %.4f \t'%(sarates[T]))
      swap_acc_rate.write('\n')

      print('      ######  iteration %.0f of pt_runs() finished at: '%(it+1) +time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

      if (it%100)==0:
         plot_mcorbits(1,it+1,100)
         plot_proj_orbits(it+1,100)
         
         T = 1
         mchain = get_mcsample(1,it+1,T,'hist')
         plot_posterior(mchain,it+1,T,True)
         mchain = []
      
      gc.collect()

   swap_acc_rate.close()



def prob_accept(data,prev_model,cand_model,var,jhr,psmin,psmax,T):

   prev_chisq_sep, prev_chisq_PA, prev_chisq_RV = prob_data_given_model(data,prev_model)
   cand_chisq_sep, cand_chisq_PA, cand_chisq_RV = prob_data_given_model(data,cand_model)

   arg = (sum(prev_chisq_sep) -sum(cand_chisq_sep) +sum(prev_chisq_PA) -sum(cand_chisq_PA) +prev_chisq_RV -cand_chisq_RV) /float(T)
   
   #if var == 3:
   #   psmin[var] = -1.
   #   psmax[var] = 1.
   #   prev_model[var] = cos(prev_model[var])
   #   cand_model[var] = cos(cand_model[var])
   
   if (not var==3):
      
      # properly normalize proposal distribution (uniform) if jump range extends beyond low end of parameter space
      if ((cand_model[var] -jhr[var]) < psmin[var]) or ((prev_model[var] -jhr[var]) < psmin[var]):
         arg += ln( (2*jhr[var]) / min((2*jhr[var]),(cand_model[var]+jhr[var]-psmin[var])) )
         arg -= ln( (2*jhr[var]) / min((2*jhr[var]),(prev_model[var]+jhr[var]-psmin[var])) )

      # properly normalize proposal distribution (uniform) if jump range extends beyond high end of parameter space
      elif ((cand_model[var] +jhr[var]) > psmax[var]) or ((prev_model[var] +jhr[var]) > psmax[var]):
         arg += ln( (2*jhr[var]) / min((2*jhr[var]),(-cand_model[var]+jhr[var]+psmax[var])) )
         arg -= ln( (2*jhr[var]) / min((2*jhr[var]),(-prev_model[var]+jhr[var]+psmax[var])) )

   if arg >= 0: prob_acc = 1.
   else: prob_acc = exp(arg)

   return prob_acc



def prob_data_given_model(data,model):   # likelihood to be expressed as function of chi-squared's

   n = len(data[0])
   JD,sep,sepunc,PA,PAunc = data[0],data[1],data[2],data[3],data[4]
   smjr_ax,P,ecc,inclin,bOmega,omega,tP = model[0],model[1],model[2],model[3],model[4],model[5],model[6]

   chisq_sep, chisq_PA = [],[]
   for i in range(n):
      modelsep, modelPA = absolute_astrometry(d,JD[i],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'sep,PA')
      chisq_sep.append( ((sep[i]-modelsep)/sepunc[i])**2 )
      chisq_PA.append( ((PA[i]-modelPA)/PAunc[i])**2 )

   # [2015-12-10]: one RV data point from Snellen et al. (2014)
   modelRV = absolute_astrometry(d,convert_gcal('2013-12-17','jdd'),smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'RV')
   chisq_RV = ((-15.4-modelRV)/1.7)**2

   return chisq_sep, chisq_PA, chisq_RV



def julian_day(jdy):
   return sum(jcal2jd(int(jdy),1,1+int(round((jdy%1)*365.25,0))))


def julian_year(jdd):
   jcal = jd2jcal(jdd,0)
   jdy = jcal[0] +(jcal[1]-1)/12. +(jcal[2]-1)/365.25
   return round(jdy,3)


def convert_gcal(gcd,jdw):
   gfmt = '%Y-%m-%d'       ### Geogorian date format in data file
   jdt = datetime.datetime.strptime(gcd,gfmt).timetuple()
   jdd = sum(gcal2jd(jdt.tm_year,jdt.tm_mon,jdt.tm_mday))
   if jdw == 'jdd': return jdd
   elif jdw == 'jdy': return julian_year(jdd)



def init_all(w81):
   
   param_list = ['semimajor axis','orbital period','eccentricity','inclination','big Omega','omega','t_periastron']
   param_ct = len(param_list)

   if w81 == False:  sdata = 1       # not using 1981-11-10 event as data point
   elif w81 == True: sdata = 2       # do use 1981-11-10 event as data point
   nchain  = 8
   swapInt = 1e4
   
   models = []
   for i in range(nchain):
      #models.append(initial_guesses(1))
      models.append(initial_guesses(''))

   #Tmin, Tmax = 1, 8
   #temp_ladder = np.arange(Tmin,Tmax+1,1).tolist()
   #logTmin,logTmax = 0,4   ### v3
   #temp_ladder = (10**(np.arange(logTmin,logTmax,0.5))).tolist()
   logTmin,logTmax = 0,5    ### v4
   temp_ladder = (10**(np.arange(logTmin,logTmax,0.71428571))).tolist()


   # jump half-ranges (uniform distribution)
   #    [2015-12-10]: switched to uniform in log(a) and cos(i)
   #    [2005-12-12]: switched back to uniform in (a)
   jhr   = [AU_to_cm(0.08), yrs_to_day(0.3), 1e-2, 5e-2, deg_to_rad(0.8), deg_to_rad(0.8), 21.]

   # define edges of parameter space
   psmin = [AU_to_cm(1),     yrs_to_day(5),   1e-9, deg_to_rad(0),   deg_to_rad(-180), deg_to_rad(-180), julian_day(1980)]
   psmax = [AU_to_cm(25),    yrs_to_day(60),  0.8,  deg_to_rad(180), deg_to_rad(180),  deg_to_rad(180),  julian_day(2015)]

   # get real data from file
   real_data = get_data(sdata)

   return param_list, param_ct, nchain, swapInt, jhr, psmin, psmax, real_data, models, temp_ladder



def initial_guesses(init):

   if init == 1:        ### MCMC result from Chauvin et al. (2012)
      return [AU_to_cm(8.8), yrs_to_day(19.6), 0.021, deg_to_rad(88.5), deg_to_rad(-148.24), deg_to_rad(-115.0), julian_day(2006.3)]
   
   elif init == 2:      ### chi-squared minimization result from Chauvin et al. (2012)
      return [AU_to_cm(11.2), yrs_to_day(28.3), 0.16, deg_to_rad(88.8), deg_to_rad(-147.73), deg_to_rad(4.0), julian_day(2013.3)]

   else:
      smjr_ax = AU_to_cm(random.uniform(5,15))
      P = yrs_to_day(random.uniform(15,25))
      ecc = random.uniform(0.001,0.1)
      inclin = arccos(random.uniform(-0.2,0.2))       ### uniform in cos(i)
      bOmega = deg_to_rad(random.uniform(-160,-140))
      omega = deg_to_rad(random.uniform(-120,-100))
      tP = random.uniform(julian_day(2005),julian_day(2010))

      return [smjr_ax,P,ecc,inclin,bOmega,omega,tP]



def mcmc_fit(swapInt,model,wnum,burnIn,verbose,param_list,param_ct,psmin,psmax,jhr,real_data,T,output,it):
   t0 = time.time()
   
   init_model = np.copy(model)
   
   acc_rate,mchain = [],[]
   for i in range(param_ct):
      acc_rate.append([])
      mchain.append([])
   
   tpsmin = np.zeros(param_ct)   # modified parameter space edge to reflect uniform priors in cos(i)
   tpsmax = np.zeros(param_ct)

   for i in range(param_ct):
      if i == 3:                      # prior for inclination is uniform in cos(i)
         tpsmin[i] = (-1.)
         tpsmax[i] = (1.)
      else:
         tpsmin[i] = (psmin[i])
         tpsmax[i] = (psmax[i])

   jump_ct = 0
   for counter in range(int(swapInt)):

      trial_val = -1e30                 # temporary placeholder value
      var = (counter%param_ct)          # which parameter (Gibbs sampler)
      
      while not (tpsmin[var] < trial_val < tpsmax[var]):  # do until new value falls within parameter space
         
         if var == 3: trial_val = cos(model[var]) +random.uniform(-jhr[var],jhr[var])
         else: trial_val = model[var] +random.uniform(-jhr[var],jhr[var])
      
      if var == 3: prop_val = arccos(trial_val)
      else: prop_val = trial_val
      
      new_model = np.append(model[:var],prop_val)
      new_model = np.append(new_model,model[var+1:])
      
      prob_acc = prob_accept(real_data,model,new_model,var,jhr,psmin,psmax,T)
      
      if np.random.rand() < prob_acc:
         model = new_model
         jump_ct += 1
      
      # [2015-12-16] burn-in reduced from 1e3 to 1e2 accepted jumps
      # [2015-12-18] burn-in reverted back to 1e3 accepted jumps, with corresponding change to 1e4 iterations between swap proposals
      if (jump_ct >= 1e3) or (burnIn == False):
         acc_rate[var].append(prob_acc)
         if (counter%1e2) == 0:
            for i in range(param_ct):
               if i==6: mchain[i].append(julian_year(model[i]))
               else: mchain[i].append(model[i])

      if ((counter%1e2)==0 and verbose) or ((counter%2e3)==0) or (counter<1e4 and (counter%1e3)==0):
         print('walker '+str(wnum+1), '%6s'%jump_ct, '%6s'%counter, '%6s'%'%.3f'%prob_acc, '%6s'%'%.2f'%(cm_to_AU(model[0])),'%6s'%'%.2f'%(day_to_yrs(model[1])),'%6s'%'%.2f'%(model[2]),'%6s'%'%.2f'%(rad_to_deg(model[3])),'%6s'%'%.2f'%(rad_to_deg(model[4])),'%6s'%'%.2f'%(rad_to_deg(model[5])),'%6s'%'%.2f'%(julian_year(model[6])))

   if True: #wnum == 0:
      print('')
      print('walker number:    '+str(wnum+1))
      print('semimajor axis : '+'median value = %6s'%'%.6f'%(cm_to_AU(np.median(mchain[0]))) \
                               +',   initial value = %6s'%'%.6f'%(cm_to_AU(init_model[0])) \
                               +',   acceptance rate = %.3f'%(np.mean(acc_rate[0])))
      print('orbital period : '+'median value = %6s'%'%.6f'%(day_to_yrs(np.median(mchain[1]))) \
                               +',   initial value = %6s'%'%.6f'%(day_to_yrs(init_model[1])) \
                               +',   acceptance rate = %.3f'%(np.mean(acc_rate[1])))
      print('eccentricity :   '+'median value = %6s'%'%.6f'%(np.median(mchain[2])) \
                               +',    initial value = %6s'%'%.6f'%(init_model[2]) \
                               +',    acceptance rate = %.3f'%(np.mean(acc_rate[2])))
      print('inclination :    '+'median value = %6s'%'%.6f'%(rad_to_deg(np.median(mchain[3]))) \
                               +',   initial value = %6s'%'%.6f'%(rad_to_deg(init_model[3])) \
                               +',   acceptance rate = %.3f'%(np.mean(acc_rate[3])))
      print('big Omega :      '+'median value = %6s'%'%.6f'%(rad_to_deg(np.median(mchain[4]))) \
                               +', initial value = %6s'%'%.6f'%(rad_to_deg(init_model[4])) \
                               +', acceptance rate = %.3f'%(np.mean(acc_rate[4])))
      print('omega :          '+'median value = %6s'%'%.6f'%(rad_to_deg(np.median(mchain[5]))) \
                               +',   initial value = %6s'%'%.6f'%(rad_to_deg(init_model[5])) \
                               +',    acceptance rate = %.3f'%(np.mean(acc_rate[5])))
      print('t_periastron :   '+'median value = %6s'%'%.6f'%(np.median(mchain[6])) \
                               +', initial value = %6s'%'%.6f'%(julian_year(init_model[6])) \
                               +', acceptance rate = %.3f'%(np.mean(acc_rate[6])))
   print('')
   print('walker number:    '+str(wnum+1))
   print('time elapsed:     %.1f seconds' % (time.time()-t0))
   print('')

   print('jump half-ranges: '+str(['%.3f AU'%(cm_to_AU(jhr[0])), '%.3f yrs'%(day_to_yrs(jhr[1])), '%.3f (eccentricity)'%(jhr[2]), '%.3e (cos i)'%(jhr[3]), '%.3f deg'%(rad_to_deg(jhr[4])), '%.3f deg'%(rad_to_deg(jhr[5])), '%.3f days'%(jhr[6])]))
   print('')


   print('      ######  saving results from (T = %.1f) chain at: '%(T)+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   
   results_list = []
   results_list.append(cm_to_AU(np.array(mchain[0])))
   results_list.append(day_to_yrs(np.array(mchain[1])))
   results_list.append((np.array(mchain[2])))
   results_list.append(rad_to_deg(np.array(mchain[3])))
   results_list.append(rad_to_deg(np.array(mchain[4])))
   results_list.append(rad_to_deg(np.array(mchain[5])))
   results_list.append((np.array(mchain[6])))

   accepted = open('accepted_orbits_logT%.1f'%(log10(T))+'_it%03d'%(it+1)+'.dat','w')
   accepted.write('# \t')
   for i in range(param_ct):
      accepted.write('  '+param_list[i]+'\t')
   accepted.write('\n')
   for k in range(len(mchain[0])):
      for i in range(param_ct):
         accepted.write(' '+'%8s'%'%.5f'%(results_list[i][k])+'\t')
      accepted.write('\n')

   for i in range(len(mchain[6])):
      mchain[6][i] = julian_day(mchain[6][i])
      
   last_posit = model
   avg_acc_rate = []
   for i in range(param_ct):
      avg_acc_rate.append('%.3f'%np.mean(acc_rate[i]))

   output.put([wnum, last_posit, avg_acc_rate])

   plot_posterior(results_list,it+1,T,False)

   print('      ######  mcmc_fit() finished for walker '+str(wnum+1)+', iteration '+str(it+1)+' at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

   gc.collect()



def absolute_astrometry(d,t,smjr_ax,P,ecc,inclin,bOmega,omega,tP,verbose,output):

   ### mean motion (n)
   n = 2 *pi() /P
   
   ### mean anomaly (M)
   M = n *(t-tP)                 ### (t-tP) gives time since periapsis

   '''
   ### solve Kepler's equation
   ###    M = E -ecc *sin(E)
   
   ### use Newton-Raphson method
   ###    f(E) = E -ecc*sin(E) -M(t)
   ### E_(n+1) = E_n - f(E_n) /f'(E_n)
   ###         = E_n - (E_n -ecc*sin(E_n) -M(t)) /(1 -ecc*cos(E_n))
   '''   

   ### eccentric anomaly (E)
   if ecc <= 0.8: E = M
   else: E = pi()
   counter = 0
   tloop = time.time()
   while np.abs(g(E,M,ecc)) > 1e-5:
      E = E - (E -ecc*sin(E) -M) /(1 -ecc*cos(E))
      counter += 1
      if verbose and counter%10 == 0: print(counter,round(time.time()-tloop,5))

   ### true anomaly (f)
   f = 2 * arctan(sqrt((1+ecc)/(1-ecc))*tan(E/2.))

   ### true separation
   r = smjr_ax *(1 -ecc**2) /(1 +ecc *cos(f))

   ### absolute astrometry
   X = r *(cos(bOmega)*cos(omega+f)-sin(bOmega)*sin(omega+f)*cos(inclin))
   Y = r *(sin(bOmega)*cos(omega+f)+cos(bOmega)*sin(omega+f)*cos(inclin))
   
   ### Kepler's third law
   M_sys = 4*pi()**2/G *smjr_ax**3 /P**2
   
   ### switch to center-of-mass frame
   CM_X = X /M_sys   #*m2/(m1 +m2)
   CM_Y = Y /M_sys   #*m2/(m1 +m2)
   
   rawarctan = arctan((Y-CM_Y)/(X-CM_X))
   if (Y-CM_Y)>=0 and (X-CM_X)>=0: PA_m2 = rawarctan  # quadrant I
   elif (X-CM_X)<0: PA_m2 = rawarctan +pi()           # quadrants II and III
   else: PA_m2 = rawarctan +2*pi()                    # quadrant IV
   
   proj_sep = sqrt((X-CM_X)**2 +(Y-CM_Y)**2)                   # physical separation in cm
   proj_sep = rad_to_deg(proj_sep/pc_to_cm(d)) *3600 *1000     # angular separation in mas

   PA = rad_to_deg(PA_m2)
   if PA >= 360: PA -= 360

   ### calculate RVs with respect to stationary barycenter (CM)
   RV_m1 = m2 /M_sys *(sec_to_day(n) *smjr_ax *sin(inclin)) /sqrt(1 -ecc**2) *(cos(omega+f) +ecc*cos(omega))
   RV_m2 = -RV_m1 *(M_sys-m2) /m2
   #RV_m1 = m2 /(m1 +m2) *(sec_to_day(n) *smjr_ax *sin(inclin)) /sqrt(1 -ecc**2) *(cos(omega+f) +ecc*cos(omega))
   #RV_m2 = -RV_m1 *m1 /m2

   if output == 'sep,PA': return proj_sep, PA
   elif output == 'RA,dec': return (rad_to_deg((Y-CM_Y)/pc_to_cm(d))*3600*1000),(rad_to_deg((X-CM_X)/pc_to_cm(d))*3600*1000),(RV_m1*1e-5),(RV_m2*1e-5)
   elif output == 'RV': return (RV_m2*1e-5)


def get_data(arg):
   
   JD, sep, sepunc, PA, PAunc = [],[],[],[],[]

   if arg == 1: data = open('data_bPicb.dat','r')
   elif arg == 2: data = open('data_bPicb_w81.dat','r')
   for line in data.readlines():
      if not line.startswith('#'):
         thisline = line.split()

         gcal = str(thisline[0])
         JD.append(convert_gcal(gcal,'jdd'))

         sep.append(float(thisline[1]))
         sepunc.append(float(thisline[2]))
         PA.append(float(thisline[3]))
         PAunc.append(float(thisline[4]))

   data.close()
   return JD, sep, sepunc, PA, PAunc



### function to be evaluated in Newton-Raphson method
def g(E,M,ecc): return E -ecc*sin(E) -M


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
def arcsin(x): return np.arcsin(x)
def arccos(x): return np.arccos(x)
def arctan(x): return np.arctan(x)
def arctan2(x,y): return np.arctan2(x,y)
def arccos(x): return np.arccos(x)
def log10(x): return np.log10(x)
def ln(x): return np.log(x)


### unit conversions
def rad_to_deg(rad): return rad *180. /pi()
def deg_to_rad(deg): return deg /180. *pi()
def sec_to_yrs(sec): return sec /60. /60 /24 /365.256362
def yrs_to_sec(yrs): return yrs *60. *60 *24 *365.256362
def day_to_yrs(day): return day /365.256362
def yrs_to_day(yrs): return yrs *365.256362
def sec_to_day(sec): return sec /60. /60 /24
def day_to_sec(day): return day *24. *60 *60
def hrs_to_sec(hrs): return hrs *60. *60.
def cm_to_AU(cm): return cm *6.68458712e-14
def AU_to_cm(AU): return AU *1.496e13
def cm_to_Rsun(cm): return cm /R_Sun
def Rsun_to_cm(Rsun): return Rsun *R_Sun
def pc_to_cm(pc): return pc *3.086e18
def g_to_Mearth(g): return g /5.972e27


### beta Pic system
d = 19.3          # pc (Hipparcos; Crifo et al. 1997)
m1 = 1.61 *M_Sun  # grams
m2 = 7.0 *M_Jup   # grams



import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,ScalarFormatter,FormatStrFormatter,NullFormatter
from pylab import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
from matplotlib import font_manager
from matplotlib import rc


def hist_post_at_temp(it,T):
   mchain = get_mcsample(1,it+1,T,'hist')
   plot_posterior(mchain,it+1,T,True)
   mchain = []
   gc.collect()


def plot_posterior(mchain,it,T,cu):
   
   fontproperties = {'family':'serif','serif':['cmr'],'weight':'normal','size':11}
   rc('text', usetex=True)
   rc('font', **fontproperties)
   plt.close()
   fig = plt.figure(1,figsize=(7.5,10.5))
   ax  = [fig.add_subplot(521),fig.add_subplot(522),fig.add_subplot(523),fig.add_subplot(524),fig.add_subplot(525),fig.add_subplot(526),fig.add_subplot(527),fig.add_subplot(529)]

   #fig = plt.figure(1,figsize=(8.0,9.0))
   #ax  = [fig.add_subplot(421),fig.add_subplot(422),fig.add_subplot(423),fig.add_subplot(424),fig.add_subplot(425),fig.add_subplot(426),fig.add_subplot(427),fig.add_subplot(428)]
   ### 8th subplot overcomes bug in matplotlib histogram (as of 2015-12-03)
   ###   see http://stackoverflow.com/questions/29791119/extra-bar-in-the-first-bin-of-a-pyplot-histogram
   ###   see also: /home/leung/coursework/ast381/term_project/code/hist_v16.pdf
   ###
   #fig = plt.figure(1,figsize=(8.0,6.6))
   #ax  = [fig.add_subplot(321),fig.add_subplot(322),fig.add_subplot(323),fig.add_subplot(324),fig.add_subplot(325),fig.add_subplot(326)]

   #majorFormatter = FormatStrFormatter('%.0f') #('%d')
   #xmajorLocator = MultipleLocator(20)
   #xminorLocator = MultipleLocator(5)

   plot_list = mchain
   xlabels = ['$a$ \ (AU)','$P$ \ (yr)','$e$','$i$ \ ($^{\circ}$)',r'$\Omega$ \ ($^{\circ}$)',r'$\omega$ \ ($^{\circ}$)','$t_{\mathrm{p}}$ \ [yr JD]']
   
   majloc = [5, 10, 0.1,  5, 5, 60, 5]
   minloc = [1,  2, 0.02, 1, 1, 10, 1]
   majfmt = ['%.0f','%.0f','%.1f','%.0f','%.0f','%.0f','%.0f']
   
   xmax = [23, 60, 0.8,  98, -138,   60, 2015]
   xmin = [ 2, 10, 0,    82, -157, -180, 2000]

   #majloc = [5, 5, 0.1,  30, 60, 60, 5]
   #minloc = [1, 1, 0.02, 10, 20, 20, 1]
   #majfmt = ['%.0f','%.0f','%.1f','%.0f','%.0f','%.0f','%.0f']
   
   #xmax = [25, 60, 0.8, 180,  180,  180, 2015]
   #xmin = [ 0,  5, 0,     0, -180, -180, 1995]

   for j in range(len(ax)-1):

      if j==3: tbins = 1200
      elif j==4: tbins = 1600
      else: tbins = 200

      ax[j].hist(plot_list[j],bins=tbins,color='green',histtype='step',alpha=0.8,lw=1.6)
      ax[j].set_xlabel(xlabels[j])
      ax[j].set_ylabel('$N_{\mathrm{orbits}}$')
      ax[j].set_xlim([xmin[j],xmax[j]])

      ax[j].xaxis.set_major_locator(MultipleLocator(majloc[j]))
      ax[j].xaxis.set_minor_locator(MultipleLocator(minloc[j]))
      ax[j].xaxis.set_major_formatter(FormatStrFormatter(majfmt[j]))

      ax[j].spines['left'].set_linewidth(0.45)
      ax[j].spines['right'].set_linewidth(0.45)
      ax[j].spines['top'].set_linewidth(0.45)
      ax[j].spines['bottom'].set_linewidth(0.25)

      if cu:
         a = plt.hist(plot_list[j],bins=200)
         if j==0: print('most probable values: ')
         print(float(a[1][np.argmax(a[0])]))
         #if j==6: print(a[1])
   
   print('')
   fig.tight_layout()

   if cu:
      plt.savefig('hist_all_logT%.1f'%(log10(T))+'_it%03d'%(it)+'.pdf')
      #plt.savefig('hist_all.pdf')
   else:
      plt.savefig('hist_logT%.1f'%(log10(T))+'_it%03d'%(it)+'.pdf')
      #plt.savefig('hist.pdf')

   plt.close()



def plot_position(param,ver):

   fontproperties = {'family':'serif','serif':['cmr'],'weight':'normal','size':14}
   rc('text', usetex=True)
   rc('font', **fontproperties)
   plt.close()
   fig = plt.figure(1,figsize=(6.0,7.0))
   ax  = [fig.add_subplot(211),fig.add_subplot(212)]

   JD_data, sep, sep_unc, PA, PA_unc = get_data(1)
   RA_data,dec_data,jdy_data,error_bar = [],[],[],[]
   for i in range(len(JD_data)):
      jdy_data.append(julian_year(JD_data[i]))
      RA_data.append(sep[i]*sin(deg_to_rad(PA[i])))
      dec_data.append(sep[i]*cos(deg_to_rad(PA[i])))
      total_sqerr = (sep[i]*deg_to_rad(PA_unc[i]))**2 +(sep_unc[i])**2
      err_bar = sqrt(0.5*total_sqerr)
      error_bar.append(err_bar)
   RV_data = -15.4
   RV_err = 1.7

   data_list = [[RA_data,dec_data],[RV_data]]
   label_list = [[r'$\Delta \alpha$ (mas)',r'$\Delta \delta$ (mas)'],'Radial velocity \ [km s$^{-1}$]']
   color_list = [['green','orange'],['blue']]

   smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(param[0]),yrs_to_day(param[1]),param[2],deg_to_rad(param[3]),deg_to_rad(param[4]),deg_to_rad(param[5]),julian_day(param[6])
   JD = np.arange(julian_day(1980),julian_day(2020),7)
   RA_list,dec_list,jdy,RV_list = [],[],[],[]
   for i in range(len(JD)):
      RA, dec, RV_m1, RV_m2 = absolute_astrometry(d,JD[i],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'RA,dec')
      RA_list.append(RA)
      dec_list.append(dec)
      RV_list.append(RV_m2)
      jdy.append(julian_year(JD[i]))
   
   plot_list = [[RA_list,dec_list],[RV_list]]

   for j in range(len(plot_list)):
      for k in range(len(plot_list[j])):
         if j==0:
            ax[j].plot(jdy,plot_list[j][k],color=color_list[j][k],marker='',ls='-',ms=1,lw=1.2,alpha=0.6,label=label_list[j][k])
         else:
            ax[j].plot(jdy,plot_list[j][k],color=color_list[j][k],marker='',ls='-',ms=1,lw=1.2,alpha=0.6,label='RV (km s$^{-1}$)') 
         ax[j].plot([1980,2020],[0,0],'k-',ls='dashed',lw=0.2)

      if j==0:
         gcal = '1981-11-10'
         jd81 = convert_gcal(gcal,'jdy')
         ax[j].scatter([jd81],[0],marker='x',edgecolor='blue',lw=2,s=36)
         for k in range(len(plot_list[j])):
            ax[j].errorbar(jdy_data,data_list[j][k],yerr=[error_bar,error_bar],fmt='none',alpha=0.6,lw=0.6,ecolor='red',capthick=0.6)
            ax[j].scatter(jdy_data,data_list[j][k],marker='o',edgecolor='red',c='red',alpha=0.8,lw=0.2,s=6)
      else:
         gcal = '2013-12-17'
         jd13 = convert_gcal(gcal,'jdy')
         ax[j].scatter([jd13],[RV_data],marker='o',edgecolor='red',c='red',alpha=0.8,lw=0.5,s=12)
         ax[j].errorbar([jd13],[RV_data],yerr=RV_err,fmt='none',alpha=0.6,lw=0.6,ecolor='red',capthick=0.6)

      ax[j].set_xlabel('JD \ [year]')
      if j == 0: ax[j].set_ylabel('Angular separation \ [mas]')
      elif j == 1: ax[j].set_ylabel(label_list[j])

      ax[j].set_xlim([1980,2020])
      ax[j].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
      ax[j].xaxis.set_major_locator(MultipleLocator(5))
      ax[j].xaxis.set_minor_locator(MultipleLocator(1))

      ax[j].spines['left'].set_linewidth(0.9)
      ax[j].spines['right'].set_linewidth(0.9)
      ax[j].spines['top'].set_linewidth(0.9)
      ax[j].spines['bottom'].set_linewidth(0.9)

      ld = ax[j].legend(loc='upper right',shadow=False,labelspacing=0.25,borderpad=0.12)
      frame = ld.get_frame()
      frame.set_lw(0)
      for label in ld.get_texts(): label.set_fontsize('medium')
      for label in ld.get_lines(): label.set_linewidth(1.2)

   fig.tight_layout()
   plt.savefig('orbit_v'+str(ver)+'.pdf')
   plt.savefig('orbit.pdf')
   plt.close()



def plot_all(mparam,ver):
   t0 = time.time()

   fontproperties = {'family':'serif','serif':['cmr'],'weight':'normal','size':14}
   rc('text', usetex=True)
   rc('font', **fontproperties)
   plt.close()
   fig = plt.figure(1,figsize=(6.0,7.0))
   ax  = [fig.add_subplot(211),fig.add_subplot(212)]

   JD_data, sep, sep_unc, PA, PA_unc = get_data(1)
   RA_data,dec_data,jdy_data,error_bar = [],[],[],[]
   for i in range(len(JD_data)):
      jdy_data.append(julian_year(JD_data[i]))
      RA_data.append(sep[i]*sin(deg_to_rad(PA[i])))
      dec_data.append(sep[i]*cos(deg_to_rad(PA[i])))
      total_sqerr = (sep[i]*deg_to_rad(PA_unc[i]))**2 +(sep_unc[i])**2
      err_bar = sqrt(0.5*total_sqerr)
      error_bar.append(err_bar)
   RV_data = -15.4
   RV_err = 1.7

   data_list = [[RA_data,dec_data],[RV_data]]
   label_list = [[r'$\Delta \alpha$ (mas)',r'$\Delta \delta$ (mas)'],'Radial velocity \ [km s$^{-1}$]']
   color_list = [['green','orange'],['blue']]

   if not (type(ver) == int):
      print('about to plot %.0f orbits' %len(mparam))
      this = 0.26
   else: this = 0.48
   
   # plot orbits
   for p in range(len(mparam)):
      if (p <= 10) or ((p%10) == 0): print(p, '%.1f'%(time.time()-t0))

      param = mparam[p]
      smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(param[0]),yrs_to_day(param[1]),param[2],deg_to_rad(param[3]),deg_to_rad(param[4]),deg_to_rad(param[5]),julian_day(param[6])
      JD = np.arange(julian_day(1980),julian_day(2025),7)
      RA_list,dec_list,jdy,RV_list = [],[],[],[]
      for i in range(len(JD)):
         RA, dec, RV_m1, RV_m2 = absolute_astrometry(d,JD[i],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'RA,dec')
         RA_list.append(RA)
         dec_list.append(dec)
         RV_list.append(RV_m2)
         jdy.append(julian_year(JD[i]))
   
      plot_list = [[RA_list,dec_list],[RV_list]]

      for j in range(len(plot_list)):
         for k in range(len(plot_list[j])):
            if j==0:
               if p==0: ax[j].plot(jdy,plot_list[j][k],color=color_list[j][k],marker='',ls='-',ms=1,lw=this,alpha=this,label=label_list[j][k])
               else: ax[j].plot(jdy,plot_list[j][k],color=color_list[j][k],marker='',ls='-',ms=1,lw=this,alpha=this)
            else:
               if p==0: ax[j].plot(jdy,plot_list[j][k],color=color_list[j][k],marker='',ls='-',ms=1,lw=this,alpha=this,label='RV (km s$^{-1}$)')
               else: ax[j].plot(jdy,plot_list[j][k],color=color_list[j][k],marker='',ls='-',ms=1,lw=this,alpha=this)
         ax[j].plot([1980,2025],[0,0],'k-',ls='dashed',lw=0.2,color='gray')

   # plot data points
   for j in range(len(plot_list)):
      if j==0:
         gcal = '1981-11-10'
         jd81 = convert_gcal(gcal,'jdy')
         ax[j].scatter([jd81],[0],marker='x',edgecolor='blue',lw=2.6,s=36)
         for k in range(len(plot_list[j])):
            ax[j].scatter(jdy_data,data_list[j][k],marker='o',edgecolor='red',c='red',lw=0.8,s=3)
            ax[j].errorbar(jdy_data,data_list[j][k],yerr=[error_bar,error_bar],fmt='none',alpha=0.6,lw=0.6,ecolor='red',capsize=1.2,capthick=0.6)
      else:
         gcal = '2013-12-17'
         jd13 = convert_gcal(gcal,'jdy')
         ax[j].scatter([jd13],[RV_data],marker='o',edgecolor='red',c='red',lw=0.8,s=12)
         ax[j].errorbar([jd13],[RV_data],yerr=RV_err,fmt='none',alpha=0.6,lw=0.6,ecolor='red',capsize=1.2,capthick=0.6)

      ax[j].set_xlabel('JD \ [year]')
      if j == 0: ax[j].set_ylabel('Angular separation \ [mas]')
      elif j == 1: ax[j].set_ylabel(label_list[j])

      ax[j].set_xlim([1980,2025])
      if j == 0: ax[j].set_ylim([-600,800])
      elif j == 1: ax[j].set_ylim([-30,30])
      ax[j].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
      ax[j].xaxis.set_major_locator(MultipleLocator(5))
      ax[j].xaxis.set_minor_locator(MultipleLocator(1))

      ax[j].spines['left'].set_linewidth(0.9)
      ax[j].spines['right'].set_linewidth(0.9)
      ax[j].spines['top'].set_linewidth(0.9)
      ax[j].spines['bottom'].set_linewidth(0.9)

      ld = ax[j].legend(loc='upper right',shadow=False,labelspacing=0.1,borderpad=0.12)
      ld.get_frame().set_lw(0)
      ld.get_frame().set_alpha(0.0)
      for label in ld.get_texts(): label.set_fontsize('small')
      for label in ld.get_lines(): label.set_linewidth(1)
      
   fig.tight_layout()
   plt.savefig('many_orbits_it'+str(ver)+'.pdf')
   plt.savefig('many_orbits.pdf')
   plt.close()



def get_mcsample(start,end,T,whatfor):

   param_list = ['semimajor axis','orbital period','eccentricity','inclination','big Omega','omega','t_periastron']
   param_ct = len(param_list)

   params = []

   if (not whatfor == 'plot'):
      for i in range(param_ct): params.append([])

      for v in range(start,end+1):
         data = open('accepted_orbits_logT%.1f'%(log10(T))+'_it%03d'%(v)+'.dat','r')
         for line in data.readlines():
            if not line.startswith('#'):
               thisline = line.split()
               for i in range(param_ct):
                  params[i].append(float(thisline[i]))
         data.close()
   
      if whatfor == 'hist':
         return params
      
      elif whatfor == 'stat':
         import scipy.stats as ss
         conf_int = []
         for i in range(param_ct):
            median = ss.scoreatpercentile(params[i],50)
            msigma = ss.scoreatpercentile(params[i],16) -median
            psigma = ss.scoreatpercentile(params[i],84) -median
            conf_int.append([param_list[i],'%6s'%'%.3f'%median,'%6s'%'%.3f'%msigma,'%6s'%'+%.3f'%psigma])
            print(conf_int[i])

         #return conf_int

   elif whatfor == 'plot':
      for v in range(start,end+1):
         data = open('accepted_orbits_logT%.1f'%(log10(T))+'_it%03d'%(v)+'.dat','r')
         for line in data.readlines():
            if not line.startswith('#'):
               thisline = line.split()
               floatparam = []
               for i in range(param_ct):
                  floatparam.append(float(thisline[i]))
               params.append(floatparam)

      return params



def prob_81nov(maxit):
   t0 = time.time()

   params = get_mcsample(1,maxit,1,'plot')
   chdate = convert_gcal('1981-11-10','jdd')
   ang_size = 0.84     # angular size of beta Pic in mas (Kervella et al. 2004)

   transit_ct = 0
   for i in range(len(params)):
      param = params[i]
      smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(param[0]),yrs_to_day(param[1]),param[2],deg_to_rad(param[3]),deg_to_rad(param[4]),deg_to_rad(param[5]),julian_day(param[6])
      modelsep, modelPA = absolute_astrometry(d,chdate,smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'sep,PA')
      if (modelsep <= 0.5*ang_size): transit_ct += 1
      if (i%1000) == 0: print(str(i)+' of '+str(len(params)), '%.1f seconds'%(time.time()-t0))

   print(transit_ct/float(len(params)))



def upcoming_transit(maxit):

   param_list = ['semimajor axis','orbital period','eccentricity','inclination','big Omega','omega','t_periastron']
   param_ct = len(param_list)

   transits = open('upcoming_transit.dat','w')
   transits.write('# \t')
   transits.write('  transit date (yr JD) \t transit date (JD) \t angular separation \t position angle \t')
   for i in range(param_ct):
      transits.write('  '+param_list[i]+'\t')
   transits.write('\n')

   params = get_mcsample(1,maxit,1,'plot')
   for i in range(len(params)):
      if (i%10)==0: print('  checking '+str(i)+' out of '+str(len(params))+' random orbits; current time is '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

      param = params[i]
      smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(param[0]),yrs_to_day(param[1]),param[2],deg_to_rad(param[3]),deg_to_rad(param[4]),deg_to_rad(param[5]),julian_day(param[6])
      dates = np.arange(julian_day(2016.5),julian_day(2019.5),0.5)
   
      for j in range(len(dates)):
         proj_sep, PA = absolute_astrometry(d,dates[j],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'sep,PA')
         if proj_sep <= 0.835/2:
            transits.write('  %.3f \t %.1f \t %.1f \t %.1f '%(julian_year(dates[j]),dates[j],proj_sep,PA))
            for k in range(len(param)):
               transits.write(' %.3f'%(param[k])+'\t')
            transits.write('\n')

   transits.close()



import scipy.stats as ss

def plot_proj_orbits(endit,howmany):

   fontproperties = {'family':'serif','serif':['cmr'],'weight':'normal','size':12}
   rc('text', usetex=True)
   rc('font', **fontproperties)
   plt.close()
   fig = plt.figure(1,figsize=(7.5,3.7))
   
   oax = fig.add_subplot(111)
   oax.spines['top'].set_color('none')
   oax.spines['left'].set_color('none')
   oax.spines['right'].set_color('none')
   oax.spines['bottom'].set_color('none')
   oax.tick_params(labelcolor='none',top='off',bottom='off',left='off',right='off')


   params = get_mcsample(1,endit,1,'hist')
   param_ct = len(params)
   most_probable, median = [],[]
   for j in range(param_ct):
      if j==3: tbins = 1200
      elif j==4: tbins = 1600
      else: tbins = 200
      a = plt.hist(params[j],bins=tbins)
      most_probable.append(float(a[1][np.argmax(a[0])]))
      median.append(ss.scoreatpercentile(params[j],50))

   ax = [fig.add_subplot(121),fig.add_subplot(122)]
   ang_size = 0.835     # angular size of beta Pic in mas (Kervella et al. 2004)
   betaPic = plt.Circle((0,0),0.5*ang_size,color='orange',alpha=0.6,lw=0)

   smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(most_probable[0]),yrs_to_day(most_probable[1]),most_probable[2],deg_to_rad(most_probable[3]),deg_to_rad(most_probable[4]),deg_to_rad(most_probable[5]),julian_day(most_probable[6])
   dates = np.arange(tP,tP+P,P/1000.)
   RA_off, dec_off = [],[]
   for j in range(len(dates)):
      RA, dec, RV_m1, RV_m2 = absolute_astrometry(d,dates[j],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'RA,dec')
      RA_off.append(RA)
      dec_off.append(dec)
   for k in range(len(ax)):
      ax[k].plot(RA_off,dec_off,color='green',marker='',ls='-',ms=1,lw=0.6,alpha=0.85,label='Most probable')

   smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(median[0]),yrs_to_day(median[1]),median[2],deg_to_rad(median[3]),deg_to_rad(median[4]),deg_to_rad(median[5]),julian_day(median[6])
   dates = np.arange(tP,tP+P,P/1000.)
   RA_off, dec_off = [],[]
   for j in range(len(dates)):
      RA, dec, RV_m1, RV_m2 = absolute_astrometry(d,dates[j],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'RA,dec')
      RA_off.append(RA)
      dec_off.append(dec)
   for k in range(len(ax)):
      ax[k].plot(RA_off,dec_off,color='magenta',marker='',ls='-',ms=1,lw=0.55,alpha=0.65,label='Median')

   params = get_mcsample(1,endit,1,'plot')
   selected = []
   for i in range(howmany):
      rsample = int(random.uniform(0,len(params)))
      selected.append(params[rsample])

   for i in range(len(selected)):
      param = selected[i]
      smjr_ax,P,ecc,inclin,bOmega,omega,tP = AU_to_cm(param[0]),yrs_to_day(param[1]),param[2],deg_to_rad(param[3]),deg_to_rad(param[4]),deg_to_rad(param[5]),julian_day(param[6])
      dates = np.arange(tP,tP+P,P/1000.)
      RA_off, dec_off = [],[]
   
      for j in range(len(dates)):
         RA, dec, RV_m1, RV_m2 = absolute_astrometry(d,dates[j],smjr_ax,P,ecc,inclin,bOmega,omega,tP,False,'RA,dec')
         RA_off.append(RA)
         dec_off.append(dec)

      #for k in range(len(ax)):
      ax[1].plot(RA_off,dec_off,color='blue',marker='',ls='-',ms=1,lw=0.12,alpha=0.16)

   JD_data, sep, sep_unc, PA, PA_unc = get_data(1)
   RA_data,dec_data,jdy_data,error_bar = [],[],[],[]
   for i in range(len(JD_data)):
      jdy_data.append(julian_year(JD_data[i]))
      RA_data.append(sep[i]*sin(deg_to_rad(PA[i])))
      dec_data.append(sep[i]*cos(deg_to_rad(PA[i])))
      total_sqerr = (sep[i]*deg_to_rad(PA_unc[i]))**2 +(sep_unc[i])**2
      err_bar = sqrt(0.5*total_sqerr)
      error_bar.append(err_bar)

   ax[0].errorbar(RA_data,dec_data,xerr=error_bar,yerr=error_bar,fmt='none',alpha=0.6,lw=0.25,ecolor='red',capsize=0.6,capthick=0.2)
   ax[0].scatter(RA_data,dec_data,marker='o',edgecolor='red',c='red',lw=0.2,s=0.8,alpha=0.6)
   
   oax.set_xlabel(r'$\Delta \alpha$ \ (mas)',fontsize='large', fontweight='bold')
   oax.set_ylabel(r'$\Delta \delta$ \ (mas)',fontsize='large', fontweight='bold')
   oax.xaxis.labelpad = 12
   oax.yaxis.labelpad = 16
   #ax.grid()
   
   axpar = [[600,-600],[16,-16]]
   aypar = [[-600,600],[-16,16]]
   axmaj = [200,4]
   axmnr = [ 50,1]

   #ax[0].scatter([0],[0],marker='+',edgecolor='orange',lw=2,s=60)
   ld = ax[0].legend(loc='upper right',shadow=False,labelspacing=0.1,borderpad=0.12)
   ld.get_frame().set_lw(0)
   ld.get_frame().set_alpha(0.0)
   for label in ld.get_texts(): label.set_fontsize('x-small')
   for label in ld.get_lines(): label.set_linewidth(1)

   for k in range(len(ax)):
      ax[k].plot([axpar[k][0],axpar[k][1]],[0,0],'k-',ls=':',lw=0.36,color='gray')
      ax[k].plot([0,0],[aypar[k][0],aypar[k][1]],'k-',ls=':',lw=0.36,color='gray')
      ax[k].set_xlim([axpar[k][0],axpar[k][1]])
      ax[k].set_ylim([aypar[k][0],aypar[k][1]])
      ax[k].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
      ax[k].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
      ax[k].xaxis.set_major_locator(MultipleLocator(axmaj[k]))
      ax[k].xaxis.set_minor_locator(MultipleLocator(axmnr[k]))
      ax[k].yaxis.set_major_locator(MultipleLocator(axmaj[k]))
      ax[k].yaxis.set_minor_locator(MultipleLocator(axmnr[k]))
      ax[k].spines['left'].set_linewidth(0.5)
      ax[k].spines['right'].set_linewidth(0.5)
      ax[k].spines['top'].set_linewidth(0.5)
      ax[k].spines['bottom'].set_linewidth(0.5)
   
   ax[1].add_patch(betaPic)
   fig.tight_layout()
   plt.savefig('projected_orbits_it1-'+str(endit)+'.pdf')
   plt.savefig('projected_orbits.pdf')
   plt.close()




def plot_mcorbits(start,end,howmany):
   t0 = time.time()

   params = get_mcsample(start,end,1,'plot')
   ssize = len(params)

   selected = []
   for i in range(howmany):
      rsample = int(random.uniform(0,ssize))
      selected.append(params[rsample])
   
   print('start plotting')
   print(len(params),len(selected))
   plot_all(selected,str(start)+'-'+str(end))
   print('done plotting, took %.1f seconds' %((time.time()-t0)))

   params,selected = [],[]
   gc.collect()


def get_params(arg):

   params = [[],[],[],[],[],[],[]]
   if arg == '': data = open('fitted_orbital_parameters.dat','r')
   elif arg == 'v1': data = open('fitted_orbital_parameters_2e5steps.dat','r')
   for line in data.readlines():
   	if not line.startswith('#'):
   		thisline = line.split()
   		for i in range(len(params)):
   			params[i].append(float(thisline[i]))
   data.close()

   print('median parameter values: ')
   print('   semimajor axis : '+'median value = %6s'%'%.3f'%(np.median(params[0]))+' AU')
   print('   orbital period : '+'median value = %6s'%'%.3f'%(np.median(params[1]))+' yrs')
   print('   eccentricity :   '+'median value = %6s'%'%.3f'%(np.median(params[2])))
   print('   inclination :    '+'median value = %6s'%'%.3f'%(np.median(params[3]))+' deg')
   print('   big Omega :      '+'median value = %6s'%'%.3f'%(np.median(params[4]))+' deg')
   print('   omega :          '+'median value = %6s'%'%.3f'%(np.median(params[5]))+' deg')
   print('   t_periastron :   '+'median value = %6s'%'%.3f'%(np.median(params[6]))+' yr JD')



def check_swap_rates():

   rates = [[],[],[],[],[],[],[],[]]
   data = open('swap_acceptance_rate.dat','r')
   for line in data.readlines():
      if not line.startswith('#'):
         thisline = line.split()
         for i in range(1,9):
            rates[i-1].append(float(thisline[i]))
   data.close()

   for i in range(len(rates)):
      print(np.mean(rates[i]))



if __name__ == '__main__':

   a = input('Enter \'hist\' or \'orb\' or \'stat\' or \'run\': ')
   temp_ladder = (10**(np.arange(0,5,0.71428571))).tolist()

   if a == 'run':
      w81 = input('Fit 1981-11-10 event as data point? (Enter boolean) ')
      t0 = time.time()
      maxit = 999
      pt_runs(maxit+1,w81)
      plot_mcorbits(1,maxit,100)
      plot_proj_orbits(maxit,100)
      for T in temp_ladder: hist_post_at_temp(maxit,T)
      prob_81nov(maxit)
      upcoming_transit(maxit)
      check_swap_rates()
      tt0 = time.time()-t0
      print('time elapsed:  %.1f hours, %.1f minutes, %.1f seconds' % ((tt0/3600.),((tt0%3600)/60.),((tt0%60))))

   else:
      lit = input('Enter last iteration: ')
      
      if a == 'hist':
         for T in temp_ladder:
            hist_post_at_temp(lit-1,T)

      elif a == 'orb':
         plot_mcorbits(1,lit-1,100)
         plot_proj_orbits(lit,100)

      elif a == 'stat':
         get_mcsample(1,lit-1,1,a)

