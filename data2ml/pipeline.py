#!usr/bin/python

# Copyright 2017, Rohan Dandage <rraadd_8@hotmail.com>
# This program is distributed under General Public License v. 3.  

"""
================================
``pipeline``
================================
"""
import sys
from os.path import exists
import pandas as pd
import numpy as np

import json

from data2ml.lib.io_ml import data2ml
from data2ml.lib.io_strs import get_logger
logging=get_logger()
import warnings
warnings.filterwarnings('ignore')

def main(cfg_fh,test=False):
	with open(cfg_fh, 'r') as f:
	    cfg = json.load(f)
	dta_fh=cfg['dta_fh']
	index=cfg['index']
	Xcols=cfg['Xcols']
	ycols=cfg['ycols']
	out_fh=cfg['out_fh']
	cores=cfg['cores']
	regORcls=cfg['regORcls']
	force=cfg['force']
	if force=='True':
		force=True
	elif force=='False':
		force=False

	dta=pd.read_csv(dta_fh,sep='\t')
	dX=dta.loc[:,[index]+Xcols].set_index(index)
	dy=dta.loc[:,[index]+ycols].set_index(index)
	for c in dy:
		dy.loc[(dy.loc[:,c]>=0),c]=1
		dy.loc[(dy.loc[:,c]<0),c]=0

	for ycol in ycols:
	    data2ml(dX=dX,dy=dy,
	        index=index,
	        Xcols=Xcols,
	        ycol=ycol,
	        out_fh=out_fh,
	        cores=cores,
	        regORcls=regORcls,
	        force=force,
	       )
	    if test:   
	    	break

if __name__ == '__main__':
    if len(sys.argv)==3:
        test=sys.argv[2]
    else:
        test=False
    main(sys.argv[1],test=test)
