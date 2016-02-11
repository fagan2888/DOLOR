import numpy as np
from astropy.io import fits
import glob
import pandas as pd

filelist = glob.glob('/Volumes/VINCE/DOLORES/night1/data/*.fts*')
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




