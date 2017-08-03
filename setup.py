#!/usr/bin/env python

# Copyright 2016, Rohan Dandage <rohan@igib.in,rraadd_8@hotmail.com>
# This program is distributed under General Public License v. 3.    


"""
========
setup.py
========

installs data2ml

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
version='0.0.2',
url='https://github.com/rraadd88/data2ml',
download_url='https://github.com/rraadd88/data2ml',
description='ml',
long_description='https://github.com/rraadd88/data2ml/README.md',
license='General Public License v. 3',
install_requires=[
                    'pandas >= 0.18.0',
                    'scipy >= 0.17.0',
                    'scikit_learn >= 0.17.1',
                    'matplotlib >= 1.5.1',
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

## JUNK ##
##make tags
#git tag -a v$(python setup.py --version) -m "beta"
#git push --tags
##rewrite old tags
#git tag new old
#git tag -d old
#git push origin :refs/tags/old
#git push --tags
#undo last commit
###git reset --hard HEAD~
## ignore tmp files
# git update-index --assume-unchanged FILE_NAME
# cd docs
# make html
# cd ../
# rsync -avv docs/_build/html docs/latest
# rsync -avv docs/_build/html docs/v1.0.0

# cd ../kc_lab_dms2dfe
# git fetch upstream
# git rebase upstream/master
# git push
# git fetch upstream --tags
# git push --tags
# cd ../dms2dfe

# cd ../kc_lab_io
# rsync -avv ../dms2dfe/docs/latest dms2dfe
# rsync -avv ../dms2dfe/docs/stable dms2dfe
# rsync -avv ../dms2dfe/docs/v1.0.0 dms2dfe
# git add --all
# git commit -m "update"
# git push
# cd ../dms2dfe

## JUNK ##