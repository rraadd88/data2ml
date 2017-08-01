#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

import sys
from os.path import exists,splitext
import argparse
#import logging
#from data2ml import configure    

from data2ml.lib.io_strs import get_logg
er  
logging=get_logging()                                    

def main():
    """
    This runs all analysis steps in tandem.

    From bash command line,

    .. code-block:: text

        python path/to/data2ml/pipeline.py path/to/project_directory
        
    :param prj_dh: path to project directory.
    
    Outputs are created in `prj_dh` in directories such as `data_lbl` , `data_fit` , `data_comparison` etc. as described in :ref:`io`.

    Optionally, In addition to envoking `pipeline`, individual programs can be accessed separately as described in :ref:`programs` section.
    Also submodules can be accessed though an API, as described in :ref:`api` section.
    Also the scripts can be envoked through bash from locally downloaded `data2ml` folder.

    """
    logging.info("start")
    parser = argparse.ArgumentParser(description='data2ml')
    parser.add_argument("prj_dh", help="path to project directory", 
                        action="store", default=False)    
    parser.add_argument("--test", help="Debug mode on", dest="test", 
                        action="store", default=False)    
    parser.add_argument("--step", help="0: configure project directory, 0.1: get molecular features, 0.2: demultiplex fastq by provided borcodes, 0.3: alignment, 1: variant calling, 2: get preferntial enrichments, 3: identify molecular determinants, 4: identify relative selection pressures, 5: make visualizations", dest="step", 
                        type=float,action="store", choices=[0,0.1,0.2,0.3,1,2,3,4,5],default=None)  
    args = parser.parse_args()
    pipeline(args.prj_dh,test=args.test,step=args.step)

def pipeline(prj_dh,step=None,test=False):
    if exists(prj_dh) :
        if step==0 or step==None:
            configure.main(prj_dh,"deps")
            configure.main(prj_dh)          
#        if step==0.1 or step==None:
#            ana0_getfeats.main(prj_dh)
#        if step==0.2 or step==None:
#            ana0_fastq2dplx.main(prj_dh)
#        if step==0.3 or step==None:
#            ana0_fastq2sbam.main(prj_dh,test)
#        if step==1 or step==None:
#            ana1_sam2mutmat.main(prj_dh)
#        if step==2 or step==None:
#            ana2_mutmat2fit.main(prj_dh,test)
#        if step==3 or step==None:
#            ana4_modeller.main(prj_dh,test)
#        if step==4 or step==None:
#            ana3_fit2comparison.main(prj_dh)
#        if step==5 or step==None:
#            ana4_plotter.main(prj_dh)
        if step==None:
            logging.info("Location of output data: %s/plots/aas/data_comparison" % (prj_dh))
            logging.info("Location of output visualizations: %s/plots/aas/" % (prj_dh))
            logging.info("For information about file formats of outputs, refer to http://kc-lab.github.io/data2ml .")
    else:
        configure.main(prj_dh)                  
    logging.shutdown()

if __name__ == '__main__':
    main()
