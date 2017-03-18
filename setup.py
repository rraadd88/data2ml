#!/usr/bin/env python

# Copyright 2016, Rohan Dandage <rohan@igib.in,rraadd_8@hotmail.com>
# This program is distributed under General Public License v. 3.    


"""
========
setup.py
========

installs dms2dfe

USAGE :
python setup.py install

Or for local installation:

python setup.py install --prefix=/your/local/dir

"""

import sys
try:
    from setuptools import setup, find_packages, Extension 
except ImportError:
    from distutils.core import setup, find_packages, Extension


if (sys.version_info[0], sys.version_info[1]) != (2, 7):
    raise RuntimeError('Python 2.7 required ')
               
# main setup command
setup(
name='data2ml',
author='Rohan Dandage',
author_email='rraadd_8@hotmail.com',
version='0.0.1',
url='https://github.com/rraadd88/data2ml',
download_url='https://github.com/rraadd88/data2ml',
description='ml',
long_description='https://github.com/rraadd88/data2ml/README.md',
license='General Public License v. 3',
install_requires=[
		    #'biopython >= 1.68',
                    'pandas >= 0.18.0',
                    'scipy >= 0.17.0',
                    'scikit_learn == 0.17.1',
                    'forestci==0.1',
                    'matplotlib >= 1.5.1',
                    'seaborn == 0.7.0',
                    #'pysam == 0.8.4',
                    #'pychimera==0.1.4',
                    #'scikit-bio==0.4.1',                    
                 ],
platforms='Tested on Ubuntu 12.04',
keywords=['bioinformatics','machine learning'],
packages=find_packages(),
package_data={'': ['data2ml/tmp','data2ml/cfg']},
include_package_data=True,
entry_points={
    'console_scripts': ['data2ml = data2ml.pipeline:main',],
    },
)
