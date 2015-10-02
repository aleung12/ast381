'''
	Andrew Leung
	UT Austin

	AST 381  Planetary Astrophysics
	Professor Adam Kraus

	Homework #2
	01 Oct 2015

'''


from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
import os
from scipy.ndimage.interpolation import shift, rotate
import matplotlib.pyplot as plt
from aplpy import FITSFigure


object_list = ['ROXs12','ROXs42B']


def gaussian_pdf(x,mu,sigma,amplitude):
	return exp(-(x-mu)**2/(2*sigma**2)) / (sigma*sqrt(2*pi())) * amplitude



### part 2: write file with (x,y) coordinates of gaussian-fitted star center
def part_2():
	
	for i in range(len(object_list)):
		
		outfile = open('star_coord_'+object_list[i]+'.dat','w')
		outfile.write('# '+'%16s'%'file name\t'\
							  +'%12s'%'x centroid\t'\
							  +'%12s'%'y centroid\t'\
							  +'%12s'%'x FWHM\t'\
							  +'%12s'%'y FWHM\n')
		if object_list[i] == 'ROXs12':
			fileindex = np.linspace(519,546,46-18)
			int_x,int_y = 611,470
		elif object_list[i] == 'ROXs42B':
			fileindex = np.append(np.linspace(472,475,75-71),np.linspace(479,485,85-78))
			fileindex = np.append(fileindex,np.linspace(487,517,117-86))
			int_x,int_y = 611,470

		fit_radius = 16

		for j in fileindex:
			os.system('rm -f '+object_list[i]+'_n0'+str(int(j))+'_drp.fits')
			
			f = fits.open('data/'+object_list[i]+'/sci/calibrated/n0'+str(int(j))+'_drp.fits.gz')

			fits.writeto(object_list[i]+'_n0'+str(int(j))+'_drp.fits',f[0].data,f[0].header)
			scidata = f[0].data
			f.close()
	
			x_inv_data = (-1*(scidata[int_y,int_x-fit_radius:int_x+fit_radius+1]))
			x_inv_data -= min(x_inv_data)
			fit_x = np.linspace(-fit_radius,fit_radius,1+2*fit_radius)
			xfitParam, xfitCovar = curve_fit(gaussian_pdf,fit_x,x_inv_data)
			
			y_inv_data = (-1*(scidata[int_y-fit_radius:int_y+fit_radius+1,int_x]))
			y_inv_data -= min(y_inv_data)
			fit_y = np.linspace(-fit_radius,fit_radius,1+2*fit_radius)
			yfitParam, yfitCovar = curve_fit(gaussian_pdf,fit_y,y_inv_data)
			
			x_offset, x_FWHM = xfitParam[0], 2*sqrt(2*ln(2))*xfitParam[1]
			y_offset, y_FWHM = yfitParam[0], 2*sqrt(2*ln(2))*yfitParam[1]

			print(object_list[i],j,x_offset,x_FWHM,y_offset,y_FWHM)
			outfile.write(object_list[i]+'_n0'+str(int(j))+'_drp.fits'+'\t'\
							  +'%8s'%'%.3f'%(float(x_offset+int_x))+'\t'\
							  +'%8s'%'%.3f'%(float(y_offset+int_y))+'\t'\
							  +'%8s'%'%.3f'%(float(x_FWHM))+'\t'\
							  +'%8s'%'%.3f'%(float(y_FWHM))+'\n')
			
			if False:	#j%5 == 0:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.spines['top'].set_color('none')
				ax.spines['left'].set_color('none')
				ax.spines['right'].set_color('none')
				ax.spines['bottom'].set_color('none')
				ax.tick_params(labelcolor='w',top='off',bottom='off',left='off',right='off')
				ax.set_xticks([])
				ax.set_yticks([])

				ax1 = fig.add_subplot(1,2,1)
				ax2 = fig.add_subplot(1,2,2)

				ax1.plot(fit_x+int_x, gaussian_pdf(fit_x+int_x,xfitParam[0]+int_x,xfitParam[1],xfitParam[2]), color='red',  marker='',ls='-', ms=1,lw=1.6)
				ax1.scatter(fit_x+int_x, x_inv_data, color='blue',  marker='o', s=1,lw=1)
				ax2.plot(fit_y+int_y, gaussian_pdf(fit_y+int_y,yfitParam[0]+int_y,yfitParam[1],yfitParam[2]), color='red',  marker='',ls='-', ms=1,lw=1.6)
				ax2.scatter(fit_y+int_y, y_inv_data, color='blue',  marker='o', s=1,lw=1)
				plt.show()
				plt.close()
		
		outfile.close()



### part 3: register images (center on star) and produce sum and median stacked images
def part_3(which_rerun):
	
	for i in range(len(object_list)):
		
		if which_rerun == 5: outfile = open(object_list[i]+'.CMsub.centered.file.list','w')
		else: outfile = open(object_list[i]+'.centered.file.list','w')
		outfile.write('# '+'%16s'%'file name\n')
						  
		infile = open('star_coord_'+object_list[i]+'.dat','r')
		
		for line in infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				filename, x_cen, y_cen = str(entries[0]),float(entries[1]),float(entries[2])
				thisshift = [512-y_cen,512-x_cen]
				
				if which_rerun == 5: thisfilename = filename[:len(filename)-8]+'CMsubtracted.fits'
				else: thisfilename = filename

				f = fits.open(thisfilename)
				scidata = f[0].data
				#print(scidata.shape)
					
				shifted = shift(scidata,thisshift)
				
				if which_rerun == 5: newfilename = filename[:len(filename)-8]+'CMsub_centered.fits'
				elif which_rerun == 6: newfilename = filename[:len(filename)-8]+'ADIsub_centered.fits'
				else: newfilename = filename[:len(filename)-8]+'centered.fits'
				os.system('rm -f '+newfilename)
				

				if which_rerun == 6: fits.writeto(newfilename,scidata,f[0].header) 
				else: fits.writeto(newfilename,shifted,f[0].header)
	
				f.close()
	
				outfile.write(newfilename+'\n')
		
		infile.close()
		outfile.close()
		
		if which_rerun == 5: infile = open(object_list[i]+'.CMsub.centered.file.list','r')
		else: infile = open(object_list[i]+'.centered.file.list','r')
		stackdata = [[]]
		linecount = 0
		for line in infile.readlines():
			if not line.startswith('#'):
				linecount += 1
				stackdata.append([])
		infile.close()
		
		counter = 0
		if which_rerun == 5: infile = open(object_list[i]+'.CMsub.centered.file.list','r')
		else: infile = open(object_list[i]+'.centered.file.list','r')
		for line in infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				h = fits.open(str(entries[0]))
				stackdata[counter] = h[0].data
				if counter+2 == linecount:
					hdrsrc = str(entries[0])
				h.close()
				counter += 1
		infile.close()
		
		h = fits.open(hdrsrc)
		
		stack = np.empty([linecount,1024,1024])
		for j in range(linecount):
			stack[j] = stackdata[j]
		#print(stack.shape,type(stack))

		median_stack = np.median(stack,axis=0)
		if which_rerun == 5: newfilename = object_list[i]+'_CMsub_centered_median.fits'
		else: newfilename = object_list[i]+'_centered_median.fits'
		os.system('rm -f '+newfilename)
		fits.writeto(newfilename,median_stack,h[0].header)
		print('wrote '+newfilename)
		
		stack_sum = np.sum(stack,axis=0)
		if which_rerun == 5: newfilename = object_list[i]+'_CMsub_centered_sum.fits'
		else: newfilename = object_list[i]+'_centered_sum.fits'
		os.system('rm -f '+newfilename)
		fits.writeto(newfilename,stack_sum,h[0].header)
		print('wrote '+newfilename)

		h.close()



### part 4: rotate images so that any planet would stack
def part_4(which_rerun):

	for i in range(len(object_list)):
		
		if which_rerun == 5: outfile = open(object_list[i]+'.CMsub.aligned.file.list','w')
		elif which_rerun == 6: outfile = open(object_list[i]+'.ADIsub.aligned.file.list','w')
		elif which_rerun == 7: outfile = open(object_list[i]+'.bPSFsub.aligned.file.list','w')
		else: outfile = open(object_list[i]+'.aligned.file.list','w')
		outfile.write('# '+'%16s'%'file name\n')
		
		if which_rerun == 5: sci_infile = open(object_list[i]+'.CMsub.centered.file.list','r')
		elif which_rerun == 6: sci_infile = open(object_list[i]+'.ADIsub.file.list','r')
		elif which_rerun == 7: sci_infile = open(object_list[i]+'.best.PSF.subtracted.file.list','r')
		else: sci_infile = open(object_list[i]+'.centered.file.list','r')
		hdr_infile = open('star_coord_'+object_list[i]+'.dat','r')
		
		scidata = []
		for line in sci_infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				filename = str(entries[0])
				f = fits.open(filename)
				scidata.append(f[0].data)
				f.close()
	
		linecount = 0
		for line in hdr_infile.readlines():
			if not line.startswith('#'):
				linecount += 1
				entries = line.split()
				filename = str(entries[0])
				g = fits.open(filename)
				thisPARANG   = g[0].header['PARANG']
				thisROTPPOSN = g[0].header['ROTPPOSN']
				thisEL       = g[0].header['EL']
				thisINSTANGL = g[0].header['INSTANGL']
				thisPA       = thisPARANG +thisROTPPOSN -thisEL -thisINSTANGL
				if linecount == 1: refPA = np.copy(thisPA)
				g.close()
					
				rotated = rotate(scidata[linecount-1],refPA-thisPA,reshape=False)
				if which_rerun == 5: newfilename = filename[:len(filename)-8]+'CMsub_aligned.fits'
				elif which_rerun == 6: newfilename = filename[:len(filename)-8]+'ADIsub_aligned.fits'
				elif which_rerun == 7: newfilename = filename[:len(filename)-8]+'bPSFsub_aligned.fits'
				else: newfilename = filename[:len(filename)-8]+'aligned.fits'
				os.system('rm -f '+newfilename)
				g = fits.open(filename)
				fits.writeto(newfilename,rotated,g[0].header)
				g.close()
				
				outfile.write(newfilename+'\n')
		
		sci_infile.close()
		hdr_infile.close()
		outfile.close()
		
		if which_rerun == 5: infile = open(object_list[i]+'.CMsub.aligned.file.list','r')
		elif which_rerun == 6: infile = open(object_list[i]+'.ADIsub.aligned.file.list','r')
		elif which_rerun == 7: infile = open(object_list[i]+'.bPSFsub.aligned.file.list','r')
		else: infile = open(object_list[i]+'.aligned.file.list','r')
		stackdata = [[]]
		linecount = 0
		for line in infile.readlines():
			if not line.startswith('#'):
				linecount += 1
				stackdata.append([])
		infile.close()
		
		counter = 0
		if which_rerun == 5: infile = open(object_list[i]+'.CMsub.aligned.file.list','r')
		elif which_rerun == 6: infile = open(object_list[i]+'.ADIsub.aligned.file.list','r')
		elif which_rerun == 7: infile = open(object_list[i]+'.bPSFsub.aligned.file.list','r')
		else: infile = open(object_list[i]+'.aligned.file.list','r')
		for line in infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				h = fits.open(str(entries[0]))
				stackdata[counter] = h[0].data
				if counter+2 == linecount:
					hdrsrc = str(entries[0])
				h.close()
				counter += 1
		infile.close()
		
		h = fits.open(hdrsrc)
		
		stack = np.empty([linecount,1024,1024])
		for j in range(linecount):
			stack[j] = stackdata[j]
		#print(stack.shape,type(stack))

		median_stack = np.median(stack,axis=0)
		if which_rerun == 5: newfilename = object_list[i]+'_CMsub_aligned_median.fits'
		elif which_rerun == 6: newfilename = object_list[i]+'_ADIsub_aligned_median.fits'
		elif which_rerun == 7: newfilename = object_list[i]+'_bPSFsub_aligned_median.fits'
		else: newfilename = object_list[i]+'_aligned_median.fits'
		os.system('rm -f '+newfilename)
		fits.writeto(newfilename,median_stack,h[0].header)
		print('wrote '+newfilename)
		
		stack_sum = np.sum(stack,axis=0)
		if which_rerun == 5: newfilename = object_list[i]+'_CMsub_aligned_sum.fits'
		elif which_rerun == 6: newfilename = object_list[i]+'_ADIsub_aligned_sum.fits'
		elif which_rerun == 7: newfilename = object_list[i]+'_bPSFsub_aligned_sum.fits'
		else: newfilename = object_list[i]+'_aligned_sum.fits'		
		os.system('rm -f '+newfilename)
		fits.writeto(newfilename,stack_sum,h[0].header)
		print('wrote '+newfilename)
		
		h.close()



def part_5():

	for i in range(len(object_list)):
		
		outfile = open(object_list[i]+'.CMsub.file.list','w')
		outfile.write('# '+'%16s'%'file name\n')

		infile = open('star_coord_'+object_list[i]+'.dat','r')
		for line in infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				filename,x_cen,y_cen = str(entries[0]),float(entries[1]),float(entries[2])

				f = fits.open(filename)
				scidata = f[0].data
				
				radial_bins = np.linspace(0,1000,1001)
				radial_displ = np.zeros([1024,1024])
				for y in range(1024):
					for x in range(1024):
						radial_displ[y,x] = int(sqrt((y-y_cen)**2+(x-x_cen)**2))

				max_radius = int(np.max(radial_bins))
				concen_radius = np.linspace(0,max_radius,max_radius+1)
				concen_median = np.zeros(max_radius+1)

				concen_values = [[]]
				for j in range(max_radius+1):
					concen_values.append([])
					
				for y in range(1024):
					for x in range(1024):
						concen_values[int(radial_displ[y,x])].append(scidata[y,x])
				for j in range(max_radius+1):
					concen_median[j] = np.median(concen_values[j])
					
				cm_pix_val = np.zeros([1024,1024])
				for y in range(1024):
					for x in range(1024):
						
						cm_pix_val[y,x] = concen_median[int(radial_displ[y,x])]
						
				newfilename = filename[:len(filename)-8]+'concen_median.fits'
				os.system('rm -f '+newfilename)
				fits.writeto(newfilename,cm_pix_val,f[0].header)
				
				CMsubtracted = scidata -cm_pix_val
				newfilename = filename[:len(filename)-8]+'CMsubtracted.fits'
				os.system('rm -f '+newfilename)
				fits.writeto(newfilename,CMsubtracted,f[0].header)
				
				f.close()
				print('wrote '+newfilename)
				outfile.write(newfilename+'\n')

		outfile.close()
		infile.close()

	part_3(5)
	part_4(5)



def part_6():

	for i in range(len(object_list)):
		
		outfile = open(object_list[i]+'.ADIsub.centered.file.list','w')
		outfile.write('# '+'%16s'%'file name\n')
		
		f = fits.open(object_list[i]+'_centered_median.fits')
		medianstack = f[0].data

		x_cen,y_cen = [],[]
		coord_file = open('star_coord_'+object_list[i]+'.dat','r')
		for line in coord_file.readlines():
			if not line.startswith('#'):
				entries = line.split()
				x_cen.append(float(entries[1]))
				y_cen.append(float(entries[2]))
		coord_file.close()
	
		infile = open(object_list[i]+'.centered.file.list','r')
		counter = 0		
		for line in infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				filename = str(entries[0])
				
				g = fits.open(filename)
				scidata = g[0].data
	
				ADIsubtracted = scidata -medianstack
				newfilename = filename[:len(filename)-13]+'ADIsubtracted.fits'
				os.system('rm -f '+newfilename)
				fits.writeto(newfilename,ADIsubtracted,g[0].header)
				
				g.close()
				print('wrote '+newfilename)
				outfile.write(newfilename+'\n')
				counter += 1	

		infile.close()
		outfile.close()
		f.close()

	part_4(6)



def part_7():

	for i in range(len(object_list)):

		outfile1 = open(object_list[i]+'.best.PSF.match.file.list','w')
		outfile1.write('# '+'%14s'%'file name\t'+'%25s'%'best match\n')
	
		outfile2 = open(object_list[i]+'.best.PSF.subtracted.file.list','w')
		outfile2.write('# '+'%16s'%'file name\n')

		infile = open('star_coord_'+object_list[i]+'.dat','r')

		for line in infile.readlines():
			if not line.startswith('#'):
				entries = line.split()
				filename, x_FWHM, y_FWHM = str(entries[0]),float(entries[3]),float(entries[4])
				thisFWHM = sqrt(x_FWHM**2+y_FWHM**2)

				if i == 0: compare_object = object_list[i+1]
				elif i == 1: compare_object = object_list[i-1]
				compare_file = open('star_coord_'+compare_object+'.dat','r')
				
				compare_FWHM,compare_file_list = [],[]
				for line in compare_file.readlines():
					if not line.startswith('#'):
						entries = line.split()
						compare_filename, x_FWHM, y_FWHM = str(entries[0]),float(entries[3]),float(entries[4])		
						compare_file_list.append(compare_filename)
						compare_FWHM.append(sqrt(x_FWHM**2+y_FWHM**2))

				best_match = compare_file_list[np.argmin(np.abs(thisFWHM-np.array(compare_FWHM)))]
				outfile1.write(filename+'\t'+best_match+'\n')
				
				f = fits.open(filename[:len(filename)-8]+'centered.fits')
				g = fits.open(best_match[:len(best_match)-8]+'centered.fits')
				subtracted = f[0].data -g[0].data

				newfilename = filename[:len(filename)-8]+'bestPSFsub.fits'			
				os.system('rm -f '+newfilename)
				fits.writeto(newfilename,subtracted,f[0].header)

				g.close(), f.close()

				outfile2.write(newfilename+'\n')

		infile.close(), outfile1.close(), outfile2.close()

	part_4(7)


def part_8():

	object_list = ['ROXs12','ROXs42B','ROXs42B']
	filename_list = ['ROXs12','ROXs42Bb','ROXs42Bc']

	for i in range(len(filename_list)):
		outfile = open('position_'+filename_list[i]+'.dat','w')
		outfile.write('# '+'%25s'%'file name\t'\
							  +'%12s'%'x planet\t'\
							  +'%12s'%'y planet\t'\
							  +'%12s'%'position angle\t'\
							  +'%12s'%'projected separation (pixels)\n')


		if filename_list[i] == 'ROXs12': int_x,int_y = 562,685
		elif filename_list[i] == 'ROXs42Bb': int_x,int_y = 629,495
		elif filename_list[i] == 'ROXs42Bc': int_x,int_y = 546,465
		parts = ['','_CMsub','_ADIsub','_bPSFsub']
		for j in parts:

			filename = object_list[i]+j+'_aligned_median.fits'
			fit_radius = 16

			f = fits.open(filename)
			scidata = f[0].data

			x_data = scidata[int_y,int_x-fit_radius:int_x+fit_radius+1]
			x_data -= min(x_data)
			fit_x = np.linspace(-fit_radius,fit_radius,1+2*fit_radius)
			xfitParam, xfitCovar = curve_fit(gaussian_pdf,fit_x,x_data)
			
			y_data = scidata[int_y-fit_radius:int_y+fit_radius+1,int_x]
			y_data -= min(y_data)
			fit_y = np.linspace(-fit_radius,fit_radius,1+2*fit_radius)
			yfitParam, yfitCovar = curve_fit(gaussian_pdf,fit_y,y_data)

			thisPARANG   = f[0].header['PARANG']
			thisROTPPOSN = f[0].header['ROTPPOSN']
			thisEL       = f[0].header['EL']
			thisINSTANGL = f[0].header['INSTANGL']
			PA_yaxis     = thisPARANG +thisROTPPOSN -thisEL -thisINSTANGL

			PA_planet    = PA_yaxis -rad_to_deg(arctan2(float(xfitParam[0]+int_x-512),float(yfitParam[0]+int_y-512)))
			if PA_planet < 0: PA_planet += 360

			outfile.write('%25s'%str(object_list[i]+j+'_aligned_median.fits')+'\t'\
					+'%8s'%'%.3f'%(float(xfitParam[0]+int_x))+'\t'\
					+'%8s'%'%.3f'%(float(yfitParam[0]+int_y))+'\t'\
					+'%8s'%'%.3f'%(float(PA_planet))+'\t'\
					+'%8s'%'%.3f'%(float(sqrt((xfitParam[0]+int_x-512)**2+(yfitParam[0]+int_y-512)**2)))+'\n')

			tf = FITSFigure(filename)
			tf.show_colorscale(cmap='gray')
			tf.add_colorbar()
			tf.colorbar.show(location='top',box_orientation='horizontal')
			plt.savefig(object_list[i]+j+'_aligned_median.pdf',dpi=200)
			plt.close()

		outfile.close()
		


### helper functions
def pi(): return np.pi
def exp(x): return np.exp(x)
def ln(x): return np.log(x)
def sqrt(x): return np.sqrt(x)
def arctan2(x,y): return np.arctan2(x,y)
def rad_to_deg(rad): return rad *180. /pi()



if __name__ == '__main__':
	#part_2()
	#part_3(0)
	#part_4(0)
	#part_5()
	#part_6()
	#part_7()
	part_8()

