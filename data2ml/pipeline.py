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
    out_fh=cfg['out_fh']
    index=cfg['index']
    Xcols=cfg['Xcols']
    ycols=cfg['ycols']

    cfg_data2ml=dict([(k,cfg[k]) for k in ['index','Xcols','cores','regORcls','force']])
    
    dta=pd.read_csv(dta_fh,sep='\t').set_index(index)
    print dta.columns
    print Xcols
    # print [c for c in Xcols if not c in dta.columns]
    # print [c for c in dta.columns if not c in Xcols]

    for ycol in ycols:
        data2ml(dX=dta,dy=dta,ycol=ycol,
            out_fh='%s.%s.pkl' % (out_fh,make_pathable_string(ycol)),
            **cfg_data2ml
           )
        if test:   
            break

if __name__ == '__main__':
    main()