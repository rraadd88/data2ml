#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_stats``
================================
"""
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.weightstats import DescrStatsW,CompareMeans
from statsmodels.sandbox.stats.multicomp import multipletests

from dms2dfe.lib.io_dfs import debad 

def testcomparison(df,smp1_cols,smp2_cols,test='ttest'):
    if len(smp1_cols)==0 or len(smp2_cols)==0:
        logging.warning("data not exist for comparison")
    else:
        col_stat='stat %s' % test
        col_pval='pval %s' % test
        df.loc[:,col_stat]=np.nan
        df.loc[:,col_pval]=np.nan
        for i in df.index:
            X=DescrStatsW(df.loc[i,smp1_cols].as_matrix())
            Y=DescrStatsW(df.loc[i,smp2_cols].as_matrix())
            if test=='ttest':
                df.loc[i,col_stat],df.loc[i,col_pval],tmp=CompareMeans(X,Y).ttest_ind()
            if test=='ztest':
                df.loc[i,col_stat],df.loc[i,col_pval]=CompareMeans(X,Y).ztest_ind()
        return df

from scipy import stats
from dms2dfe.lib.io_dfs import denan
def get_r2(data,xcol,ycol,log=None):
    data=denan(data.loc[:,[xcol,ycol]],axis='rows',condi='any')
    if len(data)!=0:
        if not log is None:
            if log==2:
                data=debad(data,axis=0,condi='any',bad=0)                
                data=np.log2(data)
                data=debad(data,axis=0,condi='any',bad='nan')
        slope, intercept, r_value, p_value, std_err = stats.linregress(data.loc[:,xcol],data.loc[:,ycol])
        return r_value
    else:
        logging.error("one/both cols are empty")
        return 0

def get_regression_metrics(y_test,y_score,
                            reg_type='lin',
                            res_rmse=True):
    from scipy.stats import linregress,spearmanr
    from sklearn.metrics import regression
    rmse=np.sqrt(regression.mean_absolute_error(y_test,y_score))
    if reg_type=='lin':
        slope, intercept, r, p_value, std_err = linregress(y_test,y_score)
        result="$r$=%0.2f" % (r)
    elif reg_type=='rank':
        r, p_value= spearmanr(y_test,y_score)
        result="$\rho$=%0.2f" % (r)
    if res_rmse:
        result="%s\nRMSE=%0.2f" % (result,rmse)        
    return result,r,rmse

from dms2dfe.lib.io_ml import denanrows
from scipy.stats import wilcoxon
from numpy import asarray,compress
def get_wilcoxon_direction(data,col_x,col_y):
    data=denanrows(data.loc[:,[col_x,col_y]])
    x=data.loc[:,col_x]
    y=data.loc[:,col_y]
    if y is None:
        d = x
    else:
        x, y = map(asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        logging.info("Warning: sample size too small for normal approximation.")
    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)
    if r_plus>r_minus: 
        return 'negative' 
    if r_minus>r_plus: 
        return 'positive'
    
def get_wilcoxon(data,col_ctrl,col_test,side='both',denan=True):
    if denan:
        data=denanrows(data.loc[:,[col_ctrl,col_test]])
    ranksum,pval=wilcoxon(data.loc[:,col_ctrl],data.loc[:,col_test],
#                           zero_method = "wilcox",
                         )
#     print "ranksum=%d; pval=%d" % (ranksum,pval)
    if side=='both':
        return pval
    else:
        pval=pval/2
        side_detected=get_wilcoxon_direction(data,col_ctrl,col_test)
        if side=='one':
            return pval,side_detected
    #         print side_detected
        else:
            if side==side_detected:
                return pval
            elif side!=side_detected:
                return 1-pval

def pval2stars(pval,ns=True,numeric=False):
    if not numeric:
        if pval < 0.0001:
            return "****"
        elif (pval < 0.001):
            return "***"
        elif (pval < 0.01):
            return "**"
        elif (pval < 0.05):
            return "*"
        else:
            if ns:
                return "ns"
            else:
                return "p = %.2g" % pval
    else:
        return "p = %.2g" % pval
