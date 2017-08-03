#!usr/bin/python

# Copyright 2017, Rohan Dandage <rraadd_8@hotmail.com>
# This program is distributed under General Public License v. 3.  

"""
================================
``pipeline``
================================
"""
import sys
import argparse

from os.path import exists
import pandas as pd
import numpy as np

import json

from data2ml.lib.io_ml import data2ml
from data2ml.lib.io_strs import make_pathable_string
from data2ml.lib.io_strs import get_logger
logging=get_logger()
import warnings
warnings.filterwarnings('ignore')

def main():
    logging.info("start")
    parser = argparse.ArgumentParser(description='data2ml')
    parser.add_argument("prj_dh", help="path to project directory", 
                        action="store", default=False)    
    parser.add_argument("--test", help="Debug mode on", dest="test", 
                        action="store", default=False)    
    args = parser.parse_args()
    pipeline(args.prj_dh,test=args.test)
    
    logging.shutdown()

def pipeline(cfg_fh,test=False):    
    with open(cfg_fh, 'r') as f:
        cfg = json.load(f)
    dta_fh=cfg['dta_fh']
    ycols=[s for s in cfg['ycols']]
    out_fh=cfg['out_fh']
    index=cfg['index']
    Xcols=cfg['Xcols']

    cfg_data2ml=dict([(k,cfg[k]) for k in ['index','Xcols','cores','regORcls','force']])
    
    dta=pd.read_csv(dta_fh,sep='\t')
    print dta.columns
    print Xcols
    # print [c for c in Xcols if not c in dta.columns]
    # print [c for c in dta.columns if not c in Xcols]
    dX=dta.loc[:,[index]+Xcols].set_index(index)
    dy=dta.loc[:,[index]+ycols].set_index(index)
    # from data2ml.lib.io_ml_metrics import y2classes

    for ycol in ycols:
        # dy=y2classes(dy,ycol,middle_percentile_skipped=0)
        data2ml(dX=dX,dy=dy,ycol=ycol,
            out_fh='%s.%s.pkl' % (out_fh,make_pathable_string(ycol)),
            **cfg_data2ml
           )
        if test:   
            break

if __name__ == '__main__':
    main()