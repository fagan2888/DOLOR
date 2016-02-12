import numpy as np
from astropy.io import fits
import glob
import pandas as pd
import os
import shutil
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize

NAXIS1 = 2100.
NAXIS2 = 2100.

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

def make_log(filelist):
	'''
	Creates a log of the observation night in the form
	of a pandas DataFrame
	'''

	cols = ['filename', 'OBS-TYPE','EXPTIME', 'TYPE', 'OBJCAT', 'SLT_ID', 'LMP_ID', 'GRM_ID']
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

	return np.around(np.median([master_bias[j] for j in range(1,len(master_bias))], axis = 0))	

def master_flats():
	all_flats = log.loc[log['TYPE'] == 'FLAT']
	grouped = all_flats.groupby('SLT_ID')
	x = np.zeros(grouped.ngroups, dtype=[('SLT_ID','a20'),('master_flat','f4',(NAXIS1,NAXIS1))])
	x['SLT_ID'] = grouped.groups.keys()
	

	for k, i in enumerate(x['SLT_ID']):                           # Iterate through groups
		single_group = grouped.get_group(i)    
		
		# Create an array containing the single flats
		array_of_flats = np.ndarray(shape=(len(single_group),NAXIS1, NAXIS2), dtype=float) 

		for i, b in enumerate(single_group['filename']):       # Iterate within the group
			data = fits.getdata(b)
			array_of_flats[i] = data
		
		master_flat = np.around(np.median([array_of_flats[j] for j in range(1,len(array_of_flats))], axis = 0))	
		x['master_flat'][k] = normalize(master_flat)
