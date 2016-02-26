import numpy as np
from astropy.io import fits
import glob
import pandas as pd
import os
import shutil
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize
from scipy.ndimage.filters import laplace, gaussian_filter, gaussian_filter1d
from scipy import interpolate
from pyraf import iraf

NAXIS1 = 2100.
NAXIS2 = 2100.

mask5_pos = [1013, 781]

data_dir = '/Volumes/VINCE/DOLORES/night1/data/'
res_dir = '/Volumes/VINCE/DOLORES/night1/results/'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

filelist = glob.glob(data_dir + '*.fts*')
_ = map(lambda x: shutil.copyfile(x, x.split('.fts')[0] + '.fits'), filelist)

filelist = glob.glob(data_dir + '*.fits*')
_ = map(lambda x: shutil.move(x, res_dir), filelist)

filelist = glob.glob(res_dir + '*.fits*')

# - - - - - - - - - - - - - - - - -- - - - -- - - - - -

def mad_based_outlier(points, thresh=3.):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def make_log(filelist):
	'''
	Creates a log of the observation night in the form
	of a pandas DataFrame
	'''

	cols = ['filename', 'OBS-TYPE','EXPTIME', 'TYPE', 'OBJCAT', 'SLT_ID', 
			'LMP_ID', 'GRM_ID']
	array = np.empty((len(filelist),len(cols)))
	log = pd.DataFrame(array, columns = cols )
	
	for i,f in enumerate(filelist):
		header = fits.getheader(f)
	
		OBSTYPE = header['OBS-TYPE'] 
		EXPTIME = header['EXPTIME']
		OBJCAT = header['OBJCAT'] 
		SLT_ID = header['SLT_ID'] 
		LMP_ID = header['LMP_ID'] 
		GRM_ID = header['GRM_ID'] 
	
		log.iloc[i] = pd.Series({'filename': f, 'OBS-TYPE': OBSTYPE,
			'EXPTIME': EXPTIME, 'OBJCAT':OBJCAT,'SLT_ID':SLT_ID,
			'LMP_ID': LMP_ID, 'GRM_ID': GRM_ID})
	
	log.loc[log['OBS-TYPE'] == 'OBJECT', 'TYPE'] = 'SCIENCE'
	log.loc[log['OBS-TYPE'] == 'ZERO', 'TYPE'] = 'BIAS'
	log.loc[log['LMP_ID'] == 'Ne+Hg+Ar+Kr', 'TYPE'] = 'ARC'
	log.loc[log['LMP_ID'] == 'Halogen', 'TYPE'] = 'FLAT'
	log.loc[log['LMP_ID'] == 'Ar', 'TYPE'] = 'Ar'

	return log

def master_bias(log):
	'''Creates the master bias from single bias frames'''
	bias = log[log['TYPE'] == 'BIAS']
	master_bias = np.ndarray(shape=(len(bias),NAXIS1, NAXIS2), dtype=float)

	for i, b in enumerate(bias['filename']):
		data = fits.getdata(b)
		master_bias[i] = data
	output = np.median([master_bias[j] for j in range(1,len(master_bias))], 
			 		   axis = 0)
	return 	np.around(output)

def master_flats(slit_id):
	all_flats = log.loc[(log['TYPE'] == 'FLAT') & (log['SLT_ID'] == slit_id)]
	grouped = all_flats.groupby('SLT_ID')
	x = np.zeros(grouped.ngroups, dtype=[('SLT_ID','a20'),
										('master_flat','f4',
										(NAXIS1,NAXIS1))])
	x['SLT_ID'] = grouped.groups.keys()
	
	for k, i in enumerate(x['SLT_ID']):                           # Iterate through groups
		single_group = grouped.get_group(i)    
		
		# Create an array containing the single flats
		array_of_flats = np.ndarray(shape=(len(single_group),NAXIS1, NAXIS2), 
									dtype=float) 

		for i, b in enumerate(single_group['filename']):       # Iterate within the group
			data = fits.getdata(b)
			array_of_flats[i] = data
		
		master_flat = np.around(np.median([array_of_flats[j] for j in range(1,len(array_of_flats))], axis = 0))	
		x['master_flat'][k] = normalize(master_flat)

def get_science(log, TYPE = 'SCIENCE', objcat = None, grism = 'LR-B'):
	if objcat is None:
		output = log[(log['TYPE'] == TYPE) & (log['GRM_ID'] == grism)]
	else:
		output = log[(log['TYPE'] == TYPE) & 
					 (log['GRM_ID'] == grism) & 
					 (log['OBJCAT'] == objcat)]
	return output

def _fits_transform(array, fname):
	hdu0 = fits.PrimaryHDU(array)
	hdulist = fits.HDUList([hdu0])
	path = '{}'.format(fname)
	hdulist.writeto(path, clobber = True)
	return path

def correct_background(data, objcat, index):
	fitsfile = _fits_transform(data, res_dir + 'tmp.fits')
	outfile = res_dir + objcat + 'bsub_{}.fits'.format(index)

	iraf.noao.imred.generic.background(input = fitsfile, 
									   output = outfile, axis = 2, 
									   naverage = 1, order = 2, low_reject = 2.0, 
									   high_reject = 1.5, niterate = 5, interactive = 'No')

log = make_log(filelist)
objcat = 'mask5'

mask5 = get_science(log, TYPE = 'SCIENCE', objcat = objcat)
arc = get_science(log, TYPE = 'ARC', objcat = objcat)

for i in range(0,len(mask5)):
	data = fits.getdata(mask5.filename.iloc[i])
	xobj, yobj = 1013, 780 
	sl = 50/2.
	data = data[yobj - sl : yobj + sl,:] 
	correct_background(data, objcat, i)

_fits_transform(data, res_dir + 'ciao.fits')


