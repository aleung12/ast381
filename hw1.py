'''
	Andrew Leung
	UT Austin

	AST 381  Planetary Astrophysics
	Professor Adam Kraus

	Homework #1
	15 Sep 2015

'''


import time
import numpy as np
import matplotlib.pyplot as plt
import coords



def orbit_predict(t,ecc,smjr_ax,P,m1,m2,inclin,omega,bOmega,T,verbose):
	
	'''
	###########################################################
	###                                                     ###
	###  orbital prediction for two-body systems            ###
	###    as a function of:                                ###
	###       time step in orbit (in HJD seconds)           ###
	###       eccentricity of orbit                         ###
	###       semimajor axis                                ###
        ###       orbital period                                ###
	###       mass of m1                                    ###
	###       mass of m2                                    ###
	###       inclination                                   ###
	###       omega, argument (or longitude) of periastron  ###
	###       big omega, longitude of ascending node        ###
	###       periastron                                    ###
	###                                                     ###
	###########################################################
	'''
	
	### Kepler's third law if orbital period is not given
	if P == '': P = kepler_3rd(m1,m2,smjr_ax)     ### P^2 \propto a^3

	### mean motion (n)
	n = 2 *pi() /P
	
	### mean anomaly (M)
	M = n *(t-T)                 ### (t-T) gives time since periapsis

	### solve Kepler's equation
	###    M = E -ecc *sin(E)
	
	### use Newton-Raphson method
	###    f(E) = E -ecc*sin(E) -M(t)
	### E_(n+1) = E_n - f(E_n) /f'(E_n)
	###         = E_n - (E_n -ecc*sin(E_n) -M(t)) /(1 -ecc*cos(E_n))
	
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
	#f = 2 * arctan2(sqrt(1+ecc)*sin(E/2.),sqrt(1-ecc)*cos(E/2.))
	f = 2 * arctan(sqrt((1+ecc)/(1-ecc))*tan(E/2.))

	if verbose:
		print('finished solving Kepler\'s equation')
		print('took '+str(counter)+' loop(s) and '+str(round(time.time()-tloop,5))+' seconds\n')
		print('      mean anomaly: '+str(rad_to_deg(M)%360)+' degrees')
		print(' eccentric anomaly: '+str(rad_to_deg(E)%360)+' degrees')
		print('      true anomaly: '+str(rad_to_deg(f)%360)+' degrees\n')

	### specific relative angular momentum
	#h = sqrt(smjr_ax*(1-ecc**2)*G*(m1+m2))
	#r = h**2/(G*(m1+m2))/(1+ecc*cos(f))

	### expected relative astrometry
	r = smjr_ax *(1 -ecc**2) /(1 +ecc *cos(f))
	x = r *cos(f)
	y = r *sin(f)

	### absolute astrometry
	X = r *(cos(bOmega)*cos(omega+f)-sin(bOmega)*sin(omega+f)*cos(inclin))
	Y = r *(sin(bOmega)*cos(omega+f)+cos(bOmega)*sin(omega+f)*cos(inclin))
	Z = r *(sin(omega+f)*sin(inclin))

	### PA calculated differently on either side of 'edge-on' inclination
	if inclin <= 0.5*pi(): PA_m2wrtm1 = 2*pi() -f
	else: PA_m2wrtm1 = f

	if verbose:
		print('      t = '+str(sec_to_day(t))+' HJD')
		print('      r = '+str(cm_to_AU(r))+' AU')
		print('      x position: '+str(cm_to_AU(x))+' AU')
		print('      y position: '+str(cm_to_AU(y))+' AU\n')
		print(' on-sky projected separation of m2 wrt m1: '+str(cm_to_AU(sqrt(X**2+Y**2)))+' AU')
		print('              position angle of m2 wrt m1: '+str(rad_to_deg(PA_m2wrtm1))+' degrees\n')

	### switch to center-of-mass frame
	CM_X = (m2*X) /(m1 +m2)
	CM_Y = (m2*Y) /(m1 +m2)

	rawarctan = arctan((Y-CM_Y)/(X-CM_X))

	### PA calculated differently on either side of 'edge-on' inclination
	if inclin <= 0.5*pi():
		if rawarctan >= 0:
			if (Y-CM_Y) >= 0: PA_m2 = 2*pi() -rawarctan
			else: PA_m2 = pi() -rawarctan		
		else:
			if (Y-CM_Y) >= 0: PA_m2 = pi() -rawarctan
			else: PA_m2 = -rawarctan
	else:
		if rawarctan <= 0:
			if (Y-CM_Y) <= 0: PA_m2 = rawarctan
			else: PA_m2 = pi() +rawarctan		
		else:
			if (Y-CM_Y) <= 0: PA_m2 = pi() +rawarctan
			else: PA_m2 = 2*pi() +rawarctan

	if PA_m2 > pi(): PA_m1 = PA_m2 -pi()
	else: PA_m1 = PA_m2 +pi()

	if verbose:
		print(' on-sky projected separation of m1 wrt CM: '+str(cm_to_AU(sqrt(CM_X**2+CM_Y**2)))+' AU')
		print(' on-sky projected separation of m2 wrt CM: '+str(cm_to_AU(sqrt((X-CM_X)**2+(Y-CM_Y)**2)))+' AU')
		print('              position angle of m2 wrt CM: '+str(rad_to_deg(PA_m2))+' degrees')
		print('              position angle of m1 wrt CM: '+str(rad_to_deg(PA_m1))+' degrees\n')

	### calculate RVs with respect to stationary barycenter (CM)
	RV_m1 = m2 /(m1 +m2) *(n *smjr_ax *sin(inclin)) /sqrt(1 -ecc**2) *(cos(omega+f) +ecc*cos(omega))
	RV_m2 = -RV_m1 *m1 /m2

	if verbose:
		print('   RV of m1 wrt CM: '+str(RV_m1*1e-5)+' km/s')
		print('   RV of m2 wrt CM: '+str(RV_m2*1e-5)+' km/s\n\n')

	return X,Y,Z,CM_X,CM_Y,PA_m1,PA_m2,RV_m1,RV_m2



### predict transit of HD 80606b and plot RV for HD 80606
def HD80606b(RV_bary,verbose):

	fig = plt.figure(1, figsize=(10,9.5))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.tick_params(labelcolor='w',top='off',bottom='off',left='off',right='off')
	ax.set_xticks([])
	ax.set_yticks([])

	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)
	
	time_int = np.linspace(2457236,2457389,3e4)
	
	X,Y,Z,CM_X,CM_Y,PA_m1,PA_m2,RV_m1,RV_m2 = [],[],[],[],[],[],[],[],[]

	for i in range(len(time_int)):
		x,y,z,c,d,e,f,g,h = orbit_predict(	### (Hebrard+10)
			day_to_sec(time_int[i]), ### time step in orbit (in HJD seconds)
			0.9330,               	 ### eccentricity of orbit
			0.455*AU,            	 ### semimajor axis
			day_to_sec(111.4367),    ### orbital period
			1.01*Sun,            	 ### mass of m1
			4.08*Jupiter,        	 ### mass of m2
			deg_to_rad(89.269),    	 ### inclination
			deg_to_rad(300.77),   	 ### omega, argument (or longitude) of periastron
			deg_to_rad(160.98),      ### big omega, longitude of ascending node (Wiktorowicz+14)
			day_to_sec(2455204.916), ### periastron
			verbose
			)
		X.append(x)
		Y.append(y)
		Z.append(z)
		CM_X.append(cm_to_AU(c))
		CM_Y.append(cm_to_AU(d))
		PA_m1.append(rad_to_deg(e))
		PA_m2.append(rad_to_deg(f))
		RV_m1.append(RV_bary +(1e-5*g))
		RV_m2.append(RV_bary +(1e-5*h))

	d = sqrt(np.array(X)**2+np.array(Y)**2)

	print('  writing out transit data file ...\n')
	data = open('HD80606_transit.dat','w')
	data.write('Time (HJD) \t Projected distance (R_Sun) \t X (R_Sun) \t Y (R_Sun) \t Z (R_Sun) \n')
	for i in range(len(d)):
		if d[i] < 1.007*R_Sun:		### transit prediction
			data.write(str(time_int[i])+'\t'+'           '+'%.6f'%(cm_to_Rsun(d[i]))+'\t'+'        '+'%.6f'%(cm_to_Rsun(X[i]))+'\t'+'%.6f'%(cm_to_Rsun(Y[i]))+'\t'+'%.6f'%(cm_to_Rsun(Z[i]))+'\n')
	data.close()

	X = cm_to_AU(np.array(X))
	Y = cm_to_AU(np.array(Y))

	print('  making plots ...\n')
	ax1.scatter(X,Y,marker='o',s=2,lw=0.01,c='red',alpha=0.65,label='m2',rasterized=True)
	ax1.scatter(CM_X,CM_Y,marker='o',s=2,lw=0.01,c='blue',alpha=0.65,label='CM',rasterized=True)

	lim = 1.02*np.max([np.max(X),np.max(Y),np.abs(np.min(X)),np.abs(np.min(Y))])
	ax1.set_xlim([-lim,lim])
	ax1.set_ylim([-lim,lim])
	ax1.set_xlabel('projected X position from m1 (AU)')
	ax1.set_ylabel('projected Y position from m1 (AU)')

	lgd1 = ax1.legend(loc='lower right',labelspacing=0.1,scatterpoints=6,shadow=0)
	frame = lgd1.get_frame() 
	frame.set_lw(0.6)
	for label in lgd1.get_texts(): label.set_fontsize('small')

	time_HJD = time_int.tolist()
	time_int = time_int-time_int[0]-0.25   ### CST is UTC-6
	tlim = [np.min(time_int),np.max(time_int)]

	ax2.plot(time_int,X,ls='-',lw=2,color='red',alpha=0.65,label='X')
	ax2.plot(time_int,Y,ls='-',lw=2,color='blue',alpha=0.65,label='Y')
	ax2.plot(time_int,CM_X,ls='-.',lw=2.5,color='red',alpha=0.65,label='CM_X')
	ax2.plot(time_int,CM_Y,ls='-.',lw=2.5,color='blue',alpha=0.65,label='CM_Y')
	ax2.set_xlim(tlim)
	ax2.set_xlabel('time (d)')
	ax2.set_ylabel('projected distance from m1 (AU)')

	lgd2 = ax2.legend(loc='lower right',labelspacing=0.1,shadow=0)
	frame = lgd2.get_frame() 
	frame.set_lw(0.6)
	for label in lgd2.get_texts(): label.set_fontsize('small')

	ax3.scatter(time_int,PA_m1,marker='o',s=2,lw=0.01,c='cyan',alpha=0.65,label='PA_m1',rasterized=True)
	ax3.scatter(time_int,PA_m2,marker='o',s=2,lw=0.01,c='red',alpha=0.65,label='PA_m2',rasterized=True)

	ax3.set_xlim(tlim)
	ax3.set_ylim([0,360])
	ax3.set_xlabel('time (d)')
	ax3.set_ylabel('position angle w.r.t. barycenter (deg)')

	lgd3 = ax3.legend(loc='lower left',labelspacing=0.1,scatterpoints=6,shadow=0)
	frame = lgd3.get_frame() 
	frame.set_lw(0.6)
	for label in lgd3.get_texts(): label.set_fontsize('small')

	ax4.scatter(time_int,RV_m1,marker='o',s=2,lw=0.01,c='cyan',alpha=0.65,label='RV_m1',rasterized=True)
	ax4.scatter(time_int,RV_m2,marker='o',s=2,lw=0.01,c='red',alpha=0.65,label='RV_m2',rasterized=True)

	ax4.set_xlim(tlim)
	ax4.set_xlabel('time (d)')
	ax4.set_ylabel('radial velocity (km/s)')

	lgd4 = ax4.legend(loc='lower left',labelspacing=0.1,scatterpoints=6,shadow=0)
	frame = lgd4.get_frame() 
	frame.set_lw(0.6)
	for label in lgd4.get_texts(): label.set_fontsize('small')	

	fig.tight_layout()
	plt.savefig('HD80606_prediction.pdf')
	plt.close()


	fig = plt.figure(1, figsize=(4.8,3.6))
	ax = fig.add_subplot(111)

	ax.plot(time_HJD,RV_m1,lw=2,color='blue',alpha=0.8,label='RV_m1')

	### match HJD to x-axis labels in CST (UTC-6)
	xticks = np.array([2457235.75, 2457266.75, 2457296.75, 2457327.75, 2457357.75, 2457388.75])
	xlabels = ['Aug 1','Sep 1','Oct 1','Nov 1','Dec 1','Jan 1']

	ax.set_xlim([xticks[0]-3,xticks[5]+3])
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels)

	ax.set_yticks(np.arange(0,5,0.2).tolist())
	ax.set_yticks(np.arange(0,5,0.05).tolist(),minor=True)
	ax.set_ylim([3.4,4.6])
	ax.set_ylabel('radial velocity (km/s)')

	fig.tight_layout()
	plt.savefig('HD80606_RV.pdf')
	plt.close()

	print('\n  minimum RV: '+str(RV_m1[RV_m1.index(min(RV_m1))])+' km/s')
	print('  extreme minimum at HJD '+str(time_HJD[RV_m1.index(min(RV_m1))])+'\n')

	print('  maxmium RV: '+str(RV_m1[RV_m1.index(max(RV_m1))])+' km/s')
	print('  extreme maximum at HJD '+str(time_HJD[RV_m1.index(max(RV_m1))])+'\n\n')



### simulate 100 randomly timed Gaia observations of HD 80606 over five years
def simulate(RV_bary,new_rand,add):

	global rand_list
	if new_rand: rand_list = np.random.rand(100)
	time_int = 2457281 + rand_list*(2459108-2457281)

	X,Y,Z,CM_X,CM_Y,PA_m1,PA_m2,RV_m1,RV_m2 = [],[],[],[],[],[],[],[],[]
	for i in range(len(time_int)):
		x,y,z,c,d,e,f,g,h = orbit_predict(	### (Hebrard+10)
			day_to_sec(time_int[i]), ### time step in orbit (in seconds)
			0.9330,               	 ### eccentricity of orbit
			0.455*AU,            	 ### semimajor axis
			day_to_sec(111.4367),    ### orbital period
			1.01*Sun,            	 ### mass of m1
			4.08*Jupiter,        	 ### mass of m2
			deg_to_rad(89.269),    	 ### inclination
			deg_to_rad(300.77),   	 ### omega, argument (or longitude) of periastron
			deg_to_rad(160.98),      ### big omega, longitude of ascending node (Wiktorowicz+14)
			day_to_sec(2455204.916), ### periastron
			0								 ### verbose?
			)
		X.append(x)
		Y.append(y)
		Z.append(z)
		CM_X.append(c)
		CM_Y.append(d)
		PA_m1.append(rad_to_deg(e))
		PA_m2.append(rad_to_deg(f))
		RV_m1.append(RV_bary +(1e-5*g))
		RV_m2.append(RV_bary +(1e-5*h))

	### go to CM frame
	X_m1 = -1*np.array(CM_X)
	Y_m1 = -1*np.array(CM_Y)

	### go to RA/Dec (microarcsec) offset from barycenter
	RA_m1  = -1*Y_m1 *1.145e-9
	DEC_m1 = X_m1 *1.145e-9

	### add astrometric standard errors for position measurements (de Bruijne+15)
	if add >= 1:
		for i in range(len(time_int)):
			RA_m1[i]  += (np.random.normal(0,5) /sqrt(2))
			DEC_m1[i] += (np.random.normal(0,5) /sqrt(2))
		
	### add parallax and associated astrometric errors (de Bruijne+15)
	if add >= 2:
		DEC_abs = []
		for i in range(len(time_int)):
			new_loc_px = coords.eq2ec((1e-6/3600*RA_m1[i])+(9.+22./60+37.5679/3600),(1e-6/3600*DEC_m1[i])+(50.+36./60+13.397/3600))
			LON_ec = new_loc_px[0]+(5.63*1e-3/3600)*sin((time_int[i]-2457286.5)*pi()/182.625)
			LAT_ec = new_loc_px[1]
			loc_eq_px = coords.ec2eq(LON_ec,LAT_ec)
			RA_m1[i]  = 1e6*3600*(loc_eq_px[0] -(9.+22./60+37.5679/3600)) +np.random.normal(0,6.7)/sqrt(2)
			DEC_m1[i] = 1e6*3600*(loc_eq_px[1] -(50.+36./60+13.397/3600)) +np.random.normal(0,6.7)/sqrt(2)
			DEC_abs.append(loc_eq_px[1])

	### add proper motion and associated astrometric errors (de Bruijne+15)
	if add >= 3:
		for i in range(len(time_int)):
			RA_m1[i]  += 45.76*1e3/yrs_to_sec(1)*cos(DEC_abs[i]) *day_to_sec(time_int[i]-2457286.5) +np.random.normal(0,3.5)/sqrt(2)
			DEC_m1[i] += 16.56*1e3/yrs_to_sec(1) *day_to_sec(time_int[i]-2457286.5) +np.random.normal(0,3.5)/sqrt(2)

	fig = plt.figure(1, figsize=(9.8,3.1))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.tick_params(labelcolor='w',top='off',bottom='off',left='off',right='off')
	ax.set_xticks([])
	ax.set_yticks([])

	ax1 = fig.add_subplot(1,3,1)
	ax2 = fig.add_subplot(1,3,2)
	ax3 = fig.add_subplot(1,3,3)

	xticks = np.array([2457280.5, 2457646.5, 2458011.5, 2458376.5, 2458741.5, 2459107.5])
	xlabels = ['Sep2015','Sep2016','Sep2017','Sep2018','Sep2019','Sep2020']

	if add < 2:
		ax1.scatter(time_int,RA_m1,marker='o',s=2,lw=0.1,c='blue',alpha=0.65)
		ax1.set_ylabel(r'RA offset ($\mu$as)')
	else:
		ax1.scatter(time_int,1e-3*RA_m1,marker='o',s=2,lw=0.1,c='blue',alpha=0.65)
		ax1.set_ylabel('RA offset (mas)')
	ax1.set_xlim([xticks[0]-30,xticks[5]+30])
	ax1.set_xticks(xticks)
	ax1.set_xticklabels(xlabels,rotation=60)
	
	if add < 2:
		ax2.scatter(time_int,DEC_m1,marker='o',s=2,lw=0.1,c='blue',alpha=0.65)
		ax2.set_ylabel(r'DEC offset ($\mu$as)')
	else:
		ax2.scatter(time_int,1e-3*DEC_m1,marker='o',s=2,lw=0.1,c='blue',alpha=0.65)
		ax2.set_ylabel('DEC offset (mas)')
	ax2.set_xlim([xticks[0]-30,xticks[5]+30])
	ax2.set_xticks(xticks)
	ax2.set_xticklabels(xlabels,rotation=60)

	if add < 2:
		ax3.scatter(RA_m1,DEC_m1,marker='o',s=2,lw=0.1,c='blue',alpha=0.65)
		ax3.set_xlabel(r'RA offset ($\mu$as)')
		ax3.set_ylabel(r'DEC offset ($\mu$as)')
	else:
		ax3.scatter(1e-3*RA_m1,1e-3*DEC_m1,marker='o',s=2,lw=0.1,c='blue',alpha=0.65)
		ax3.set_xlabel('RA offset (mas)')
		ax3.set_ylabel('DEC offset (mas)')
	ax3.xaxis.labelpad = 12

	fig.tight_layout()
	if add == 0:   plt.savefig('simulate_no_noise.pdf')
	elif add == 1: plt.savefig('simulate_noise_and_planet_only.pdf')
	elif add == 2: plt.savefig('simulate_with_parallax.pdf')
	elif add == 3: plt.savefig('simulate_with_px_and_pm.pdf')
	plt.close()



### define Kepler's third law
def kepler_3rd(m1,m2,a):
	
	P = ((4 *pi()**2) /(G *(m1 +m2)) *a**3 )**0.5
	return P


### function to be evaluated in Newton-Raphson method
def g(E,M,ecc): return E -ecc*sin(E) -M


### helper functions
def pi(): return np.pi
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
def cm_to_Rsun(cm): return cm /R_Sun



### define Newton's gravitational constant
G = 6.67259e-8						### in cgs [cm^3 g^-1 s^-2]



### known values with which to test
Sun     = 1.989e33      ### grams
Earth   = 5.972e27      ### grams
Jupiter = 1.898e30      ### grams
Pluto   = 1.309e22      ### grams
AU      = 1.496e13      ### cm
R_Sun   = 6.955e10      ### cm
ecc_Earth = 0.01671123
ecc_Pluto = 0.24897



### function generates diagnostic plots
def plot_motion(verbose):

	fig = plt.figure(1, figsize=(10,9.5))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.tick_params(labelcolor='w',top='off',bottom='off',left='off',right='off')
	ax.set_xticks([])
	ax.set_yticks([])

	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)
	
	time_int = np.linspace(-100,300,5e2)
	
	X,Y,Z,CM_X,CM_Y,PA_m1,PA_m2,RV_m1,RV_m2 = [],[],[],[],[],[],[],[],[]

	for i in range(len(time_int)):
		a,b,z,c,d,e,f,g,h = orbit_predict(
			yrs_to_sec(time_int[i]), ### time step in orbit
			ecc_Pluto,               ### eccentricity of orbit
			39.5*AU,                 ### semimajor axis
			'',
			Sun,                     ### mass of m1
			0.25*Sun,                ### mass of m2
			deg_to_rad(120),         ### inclination
			deg_to_rad(120),         ### omega, argument (or longitude) of periastron
			deg_to_rad(0),           ### big omega, longitude of ascending node
			0,                       ### time of periapsis
			verbose
			)
		X.append(cm_to_AU(a))
		Y.append(cm_to_AU(b))
		CM_X.append(cm_to_AU(c))
		CM_Y.append(cm_to_AU(d))
		PA_m1.append(rad_to_deg(e))
		PA_m2.append(rad_to_deg(f))
		RV_m1.append(1e-5*g)
		RV_m2.append(1e-5*h)

	time_int = sec_to_yrs(time_int)

	ax1.scatter(X,Y,marker='o',s=2,lw=0.01,c='red',alpha=0.65,label='m2')
	ax1.scatter(CM_X,CM_Y,marker='o',s=2,lw=0.01,c='blue',alpha=0.65,label='CM')

	lim = 1.02*np.max([np.max(X),np.max(Y),np.abs(np.min(X)),np.abs(np.min(Y))])
	ax1.set_xlim([-lim,lim])
	ax1.set_ylim([-lim,lim])
	ax1.set_xlabel('projected X position from m1 (AU)')
	ax1.set_ylabel('projected Y position from m1 (AU)')

	lgd1 = ax1.legend(loc='lower right',labelspacing=0.1,shadow=0)
	frame = lgd1.get_frame() 
	frame.set_lw(0.6)
	for label in lgd1.get_texts(): label.set_fontsize('small')

	ax2.plot(time_int,X,ls='-',lw=2,color='red',alpha=0.65,label='X')
	ax2.plot(time_int,Y,ls='-',lw=2,color='blue',alpha=0.65,label='Y')
	ax2.plot(time_int,CM_X,ls='-.',lw=2.5,color='red',alpha=0.65,label='CM_X')
	ax2.plot(time_int,CM_Y,ls='-.',lw=2.5,color='blue',alpha=0.65,label='CM_Y')

	tlim = [np.min(time_int),np.max(time_int)]

	ax2.set_xlim(tlim)
	ax2.set_xlabel('time (yr)')
	ax2.set_ylabel('projected distance from m1 (AU)')

	lgd2 = ax2.legend(loc='lower right',labelspacing=0.1,shadow=0)
	frame = lgd2.get_frame() 
	frame.set_lw(0.6)
	for label in lgd2.get_texts(): label.set_fontsize('small')

	ax3.scatter(time_int,PA_m1,marker='o',s=2,lw=0.01,c='cyan',alpha=0.65,label='PA_m1')
	ax3.scatter(time_int,PA_m2,marker='o',s=2,lw=0.01,c='red',alpha=0.65,label='PA_m2')

	ax3.set_xlim(tlim)
	ax3.set_ylim([0,360])
	ax3.set_xlabel('time (yr)')
	ax3.set_ylabel('position angle w.r.t. barycenter (deg)')

	lgd3 = ax3.legend(loc='lower left',labelspacing=0.1,scatterpoints=6,shadow=0)
	frame = lgd3.get_frame() 
	frame.set_lw(0.6)
	for label in lgd3.get_texts(): label.set_fontsize('small')

	ax4.scatter(time_int,RV_m1,marker='o',s=2,lw=0.01,c='cyan',alpha=0.65,label='RV_m1')
	ax4.scatter(time_int,RV_m2,marker='o',s=2,lw=0.01,c='red',alpha=0.65,label='RV_m2')

	ax4.set_xlim(tlim)
	ax4.set_xlabel('time (yr)')
	ax4.set_ylabel('radial velocity (km/s)')

	lgd4 = ax4.legend(loc='lower left',labelspacing=0.1,scatterpoints=6,shadow=0)
	frame = lgd4.get_frame() 
	frame.set_lw(0.6)
	for label in lgd4.get_texts(): label.set_fontsize('small')	

	fig.tight_layout()
	plt.show()

	plt.close()



### if run from command line
if __name__ == '__main__':
	HD80606b(3.7888,1)
	simulate(3.7888,1,0)
	simulate(3.7888,0,1)
	simulate(3.7888,0,2)
	simulate(3.7888,0,3)
