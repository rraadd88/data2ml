#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_ml_data``
================================
"""
from os.path import abspath,dirname,exists,basename
from os import makedirs

from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,regression

from dms2dfe.lib.io_data_files import read_pkl,to_pkl
from dms2dfe.lib.io_dfs import set_index,denan,denanrows,del_Unnamed
from dms2dfe.lib.io_nums import is_numeric
from dms2dfe.lib.io_plots import saveplot,get_axlims

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # no Xwindows

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
from dms2dfe.lib.io_strs import get_logger
logging=get_logger()
# from dms2dfe.lib.io_strs import get_time
# logging.basicConfig(format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s',level=logging.DEBUG) # filename=cfg_xls_fh+'.log'

def y2classes(data_combo,y_coln,classes=2,
             middle_percentile_skipped=0):
    data_combo.loc[:,'classes']=np.nan
    if classes==2:
        median=data_combo.loc[:,y_coln].median()
        if middle_percentile_skipped==0:            
            data_combo.loc[data_combo.loc[:,y_coln]>=median,"classes"]="high"
            data_combo.loc[data_combo.loc[:,y_coln]<median,"classes"]="low"
        else:
            up_bound=data_combo.loc[~pd.isnull(data_combo.loc[:,y_coln]),y_coln].quantile(0.5+middle_percentile_skipped/2)            
            lw_bound=data_combo.loc[~pd.isnull(data_combo.loc[:,y_coln]),y_coln].quantile(0.5-middle_percentile_skipped/2)
            # print up_bound
            # print lw_bound            
            data_combo.loc[data_combo.loc[:,y_coln]>up_bound,"classes"]="high"
            data_combo.loc[data_combo.loc[:,y_coln]<lw_bound,"classes"]="low"
    return data_combo

def X_cols2numeric(data_all,X_cols,keep_cols=[]):
    """
    This converts features in text form (eg. C, H, L, ..) to numeric (eg. 0, 1, 2, ..)
    
    :param data_all: dataframe with all Xs in columns.
    :param X_cols: name/s of colum/s with feature/s
    :returns data_all: dataframe with all numeric Xs.
    """
    for X_col in X_cols:
        # if not data_all.applymap(np.isreal).all(0)[X_col]:
        if not is_numeric(data_all.loc[:,X_col]):
            if not X_col in keep_cols:
                le = LabelEncoder()
                le.fit(data_all.loc[:,X_col])            
                data_all.loc[:,X_col]=le.transform(data_all.loc[:,X_col])
    return data_all

def X_cols2binary(data,cols=None):    
    if cols==None:
        cols=[]
        for col in data.columns.tolist():
            if not is_numeric(data.loc[:,col]):
                cols.append(col)
    for col in cols:
        classes=list(data.loc[:,col].unique())
        if np.nan in classes:
            classes.remove(np.nan)
        for classi in classes:            
            data.loc[data.loc[:,col]==classi,"%s: %s" % (col,classi)]=1
            data.loc[~(data.loc[:,col]==classi),"%s: %s" % (col,classi)]=0
            data.loc[(data.loc[:,col]==np.nan),"%s: %s" % (col,classi)]=np.nan
        data=data.drop(col,axis=1)
    return data

def zscore(df,col):
    df.loc[:,col] = (df.loc[:,col]-df.loc[~pd.isnull(df.loc[:,col]),col].mean())/df.loc[~pd.isnull(df.loc[:,col]),col].std()
    return df

def rescalecols(data_combo,kind="zscore"):
    for col in data_combo.columns.tolist():
#         print col
        if is_numeric(data_combo.loc[:,col]):
            if kind=="zscore" and data_combo.loc[~pd.isnull(data_combo.loc[:,col]),col].std() !=0:
#                 print col
                data_combo=zscore(data_combo,col)
            else:
                data_combo.loc[:,col]=data_combo.loc[:,col]
    return data_combo

def binary2classes(y_pred,classes):
    y_pred_classes=[]
    if len(classes)>2:
        for row in y_pred:
            for classi in range(len(classes)):
                if np.sum(row)==1:
                    if row[classi]==1:
                        y_pred_classes.append(classes[classi])
                        break
                else:
                    y_pred_classes.append(np.nan)
                    break
    elif len(classes)==2:
        for row in y_pred:
            for classi in range(len(classes)):
                if row==0:
                    y_pred_classes.append(classes[classi])
                    break
                elif row==1:
                    y_pred_classes.append(classes[classi])
                    break
    return y_pred_classes

# diff
def get_abs_diff(mutid_mm,data_feats,col):
    mutid_m1,mutid_m2=mutid_mm.split(':')
    return abs(data_feats.loc[mutid_m1,col]-data_feats.loc[mutid_m2,col])

def add_feats(data_feats_all_mm,data_feats_all):
    for col in data_feats_all:
        data_feats_all_mm.loc[:,"$\Delta$(%s)" % col]=data_feats_all_mm.apply(lambda x: \
                                    get_abs_diff(x['mutids'],data_feats_all,col), axis=1)
    return data_feats_all_mm


def get_cols_del(data_feats):
    cols_del_strs=["Helix formation","beta bridge","Chirality","Offset from residue to the partner",
             "Energy (kcal/mol) of ","Secondary structure",
             'torsion','pKa',
             'cosine of the angle between C=O of residue and C=O of previous residue',
             '$\Delta$(Molecular Polarizability) per substitution',
             '$\Delta$(Molecular weight (Da)) per substitution',
             '$\Delta$(Molecular Refractivity) per substitution',
             'bend',' coordinates of ','(Zyggregator)','(NORSnet)','(PROFbval)','(Ucon)',
              'Residue (C-alpha) depth',]
    cols_del=[]
    for c in cols_del_strs:
        cols_del=cols_del+[col for col in data_feats if c in col]
    return cols_del

def keep_cols(dXy,dXy_ori,ycol,cols_keep=None):
    if cols_keep is None:
        cols_keep=[
         'Conservation score (inverse shannon uncertainty): gaps ignored',#'Conservation score (ConSurf)',
         '$\\Delta\\Delta G$ per mutation',
         '$\\Delta$(logP) per substitution',
         'Distance from active site residue: minimum',
         '$\\Delta$(Solubility (%)) per substitution',
         '$\\Delta$(Polar Surface Area) per substitution',
         'Distance from dimer interface',
         'Temperature factor (flexibility)',
         '$\\Delta$(Solvent Accessible Surface Area) per substitution',
         'Residue depth']
    for c in cols_keep:
        if not c in dXy:
            if c in dXy_ori:
                dXy.loc[:,c]=dXy_ori.loc[:,c]
    return make_dXy(dXy,ycol,unique_quantile=0,index="mutids",if_rescalecols=False)

def get_corr_feats(corr,mx=0.9):
    feats=[]
    for row in corr.index:
        for col in corr.columns:
            if row!=col:
                if not col in feats:
                    if (corr.loc[row,col]>mx):
                        feats.append(row)
    return np.unique(feats)

def feats_sel_corr(dXy,ycol,method='spearman',range_coef=[0.9,0.8,0.7]):
    Xcols=[c for c in dXy.columns.tolist() if c!=ycol]
    data=dXy.loc[:,Xcols]
    del_cols=data.columns.tolist()
    for mx in range_coef:
        while len(del_cols)!=0:
            if len(del_cols)!=len(data.columns):
                data=data.drop(del_cols,axis=1)
            logging.info( '%s' % len(del_cols))
            del_cols=get_corr_feats(data.corr(method=method),mx=mx)
        del_cols=data.columns.tolist()
    Xcols=data.columns.tolist()
    dXy=dXy.loc[:,Xcols+[ycol]]
    return dXy,Xcols,ycol

def feats_inter(dXy,ycol,cols1=None,cols2=None,
                inter='all',
                if_rescalecols=True,
                join=True):
    Xcols=[c for c in dXy.columns.tolist() if c!=ycol]
    if cols1 is None and cols2 is None:
        cols_predefined=False
    else:
        cols_predefined=True
    if not cols_predefined:
        cols1=Xcols
        cols2=Xcols
    if if_rescalecols:
        cols = cols1 + [i for i in cols1 if i not in cols2]
        d=rescalecols(dXy.loc[:,cols], kind='zscore')
    dinter=pd.DataFrame(index=d.index)
    for c1i,c1 in enumerate(cols1):
        for c2i,c2 in enumerate(cols2):
            if c1!=c2:
                if len([c for c in dinter if ((c1 in c) and (c2 in c))])==0:
                    # print '(%s) - (%s)' % (c1,c2)
                    if inter=='-' or inter=='all':
                        dinter.loc[:,'(%s) - (%s)' % (c1,c2)]=d.loc[:,c1].sub(d.loc[:,c2])
                    if inter=='*' or inter=='all':
                        dinter.loc[:,'(%s) * (%s)' % (c1,c2)]=d.loc[:,c1].mul(d.loc[:,c2])
                    if inter=='/' or inter=='all':
                        dinter.loc[:,'(%s) / (%s)' % (c1,c2)]=d.loc[:,c1].div(d.loc[:,c2])
    if join:
        dXy=dXy.join(dinter)
    else:
        dXy=dinter
    return dXy,[c for c in dXy.columns.tolist() if c!=ycol],ycol

def feats_inter_sel_corr(dXy,ycol,Xcols,dXy_input,top_cols=None,range_coef=[0.9,0.8,0.7]):
    dXy,Xcols,ycol=feats_inter(dXy,ycol,cols1=top_cols,cols2=top_cols)
    dXy,Xcols,ycol=keep_cols(dXy,dXy_input,ycol,cols_keep=dXy_input.columns.tolist())
    dXy,Xcols,ycol=feats_sel_corr(dXy,ycol,range_coef=range_coef)
    dXy,Xcols,ycol=keep_cols(dXy,dXy_input,ycol,cols_keep=dXy_input.columns.tolist())
    return dXy,Xcols,ycol

def make_dXy(dXy,ycol,unique_quantile=0.25,index="mutids",if_rescalecols=True):
    dXy=set_index(dXy,index)
    # print 'len(cols_del)=%s' % len(get_cols_del(dXy))
    dXy=dXy.drop(get_cols_del(dXy),axis=1)
    Xcols=[c for c in dXy.columns.tolist() if c!=ycol]
    Xunique=pd.DataFrame({'unique':[len(np.unique(dXy[c])) for c in Xcols]},index=[c for c in Xcols])
    Xcols=Xunique.index[Xunique['unique']>Xunique['unique'].quantile(unique_quantile)]
    dXy=dXy.loc[:,Xcols.tolist()+[ycol]]
    dXy=dXy.dropna(axis=1, how='all').dropna(axis=0, how='any')
    if if_rescalecols:
        Xcols=[c for c in dXy.columns.tolist() if c!=ycol]
        dXy.loc[:,Xcols]=rescalecols(dXy.loc[:,Xcols])
    return dXy,Xcols,ycol

# from boruta import BorutaPy
# def feats_sel_boruta(model,dXy,Xcols,ycol):
#     model_boruta = BorutaPy(model, n_estimators='auto', random_state=88)
#     X=dXy.loc[:,Xcols].as_matrix()
#     y=dXy.loc[:,ycol].as_matrix()
#     model_boruta.fit(X,y)
# #     print Xcols,model_boruta.support_
#     Xcols=np.array(Xcols)
#     Xcols=Xcols[np.array(model_boruta.support_)]
#     return dXy.loc[:,Xcols.tolist()+[ycol]],Xcols,ycol

# def make_input(d,ycol,index="mutids",if_rescalecols=True):
#     d=set_index(d,index)
#     # remove feats with unique categories deviating max by 1sd
#     if if_rescalecols:
#         d=rescalecols(d)
#     d=d.dropna(axis=0, how='any').dropna(axis=1, how='all')
#     return d

def make_cls_input(data_combo,y_coln_cls,middle_percentile_skipped):
    data_ml=y2classes(data_combo,y_coln_cls,
                       middle_percentile_skipped=middle_percentile_skipped)
    data_ml=data_ml.drop(y_coln_cls,axis=1)
    X_cols_cls=data_ml.columns.tolist()
    y_coln_cls="classes"
    X_cols_cls=data_ml.columns.tolist().remove(y_coln_cls)
    data_ml_mutids=list(data_ml.index)
    # print sum(~pd.isnull(data_combo.loc[:,y_coln_cls]))
    data_combo=set_index(data_combo,"mutids")
    data_ml=set_index(data_ml,"mutids")
    # data_feats=set_index(data_feats,"mutids")
    # data_combo=pd.concat([data_feats,
    #                       data_combo.loc[:,y_coln_cls]],axis=1)
    # print sum(~pd.isnull(data_combo.loc[:,y_coln_cls]))
    # data_combo.index.name='mutids'
    y=data_ml.loc[:,y_coln_cls]
    data_ml=X_cols2binary(data_ml.drop(y_coln_cls,axis=1))
    data_ml.loc[:,y_coln_cls]=y#60data_ml.loc[:,y_coln_cls]
    data_ml=rescalecols(data_ml)
    data_cls_train=denan(data_ml,axis='both',condi='all any')
    # print sum(~pd.isnull(data_ml.loc[:,y_coln_cls]))

    data_cls_train_mutids=list(data_cls_train.index.values)
    data_cls_tests_mutids=[mutid for mutid in data_ml_mutids if not mutid in data_cls_train_mutids]
    data_cls_tests=data_ml.loc[data_cls_tests_mutids,:]
    return data_combo,data_ml,data_cls_train,data_cls_tests

def make_reg_input(data_combo,data_cls_train,data_cls_tests,
                    feature_importances_cls,
                    y_coln_reg,
                    y_coln_cls="classes",
                    topNfeats=25
                    ):
    data_reg_train=data_cls_train.copy()
    data_reg_tests=data_cls_tests.copy()
    data_reg_train.loc[:,y_coln_reg]=data_combo.loc[data_cls_train.index.values,y_coln_reg]
    data_reg_tests.loc[:,y_coln_reg]=data_combo.loc[data_reg_tests.index.values,y_coln_reg]
    if y_coln_cls in data_reg_train.columns.tolist(): 
        data_reg_tests=data_reg_train.drop(y_coln_cls,axis=1)
    if y_coln_cls in data_reg_tests.columns.tolist(): 
        data_reg_tests=data_reg_tests.drop(y_coln_cls,axis=1)
    # data_reg_train=denan(data_reg_train,axis='both',condi='all any')
    # data_reg_tests=denan(data_reg_tests,axis='both',condi='all any')
    # data_reg_train=X_cols2binary(data_reg_train,[y_coln_cls])
    # data_reg_tests=X_cols2binary(data_reg_tests,[y_coln_cls])
    X_cols_top=feature_importances_cls.sort_values(by='Importance',ascending=False).\
                                            head(topNfeats).loc[:,'Feature'].tolist()
    X_cols_reg=[col for col in X_cols_top\
                    if col in data_reg_tests.columns.tolist()]
    data_reg_train=data_reg_train.loc[:,X_cols_reg+[y_coln_reg]]
    data_reg_tests=data_reg_tests.loc[:,X_cols_reg+[y_coln_reg]]
    return data_reg_train,data_reg_tests


def make_data_combo(data_fit_dm,data_feats,ycol,Xcols):
    cols_numeric=[c for c in data_feats if is_numeric(data_feats.loc[:,c])]
    data_feats=data_feats.loc[:,cols_numeric]
    for col in get_cols_del(data_feats):
        del data_feats[col]
    data_feats_dm=pd.DataFrame(index=data_fit_dm.index)
    data_feats_dm.loc[:,ycol]=data_fit_dm.loc[:,ycol]
    data_feats_dm=add_feats(data_feats_dm.reset_index(),data_feats)
    return data_feats_dm
