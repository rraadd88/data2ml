#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_ml``
================================
"""
from os.path import abspath,dirname,exists,basename
from os import makedirs

from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.metrics import roc_curve, auc
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
# logging.basicConfig(format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s',level=logging.DEBUG) # filename=cfg_xls_fh+'.log'



def run_RF_classi(data_all,X_cols,y_coln,
           test_size=0.34,data_test=None,data_out_fh=None):
    """
    This implements Random Forest classifier.
    
    :param data_all: dataframe with columns with features(Xs) and classes(y).
    :param X_cols: list of column names with features.
    :param y_coln: column name of column with classes.
    :param plot_fh: path to output plot file. 
    :returns grid_search: trained classifier object.
    :returns y_test: classes used for testing classifier. 
    :returns y_pred: predicted classes.
    :returns y_score: scores of predicted classes used to plot ROC curve.
    :returns feature_importances: relative importances of features (dataframe).
    """
    from sklearn.ensemble import RandomForestClassifier

    X=data_all.loc[:,list(X_cols)]
    X=X.as_matrix()

    y=data_all.loc[:,y_coln]
    classes=y.unique()
    y=y.as_matrix()
    y = label_binarize(y, classes=classes)
    if len(classes)==2:
        y=np.array([i[0] for i in y])

    if len(classes)>1:
        if test_size!=0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=88)
        else :
            X_train=X
            y_train=y
            X_test_df=data_test.loc[:,list(X_cols)]
            X_test_df=denan(X_test_df,axis='both',condi='all any')
            X_test=X_test_df.as_matrix()
            y_test=None

        model = RandomForestClassifier(random_state =88)
        param_grid = {"n_estimators": [1000],
                      "max_features": ['sqrt'],#[None,'sqrt','log2'],
                      "min_samples_leaf":[1],#[1,25,50,100],
                      "criterion": ['entropy'],#["gini", "entropy"]
                     }

        grid_search = GridSearchCV(model, param_grid=param_grid,cv=10)
        grid_search.fit(X_train,y_train)

        y_pred=grid_search.predict(X_test)
        if test_size!=0:    
            data_preds=None
        else:
            data_preds=X_test_df
            data_preds[y_coln]=binary2classes(y_pred,classes)

    featimps=pd.DataFrame(columns=['Feature','Importance'])
    featimps.loc[:,'Feature']=X_cols#[indices]
    featimps.loc[:,'Importance']=grid_search.best_estimator_.feature_importances_

    data={'RF_classi':grid_search,
          'X_train':X_train,
          'X_test':X_test,
          'y_train':y_train,
          'y_test':y_test,
          'y_score':grid_search.predict_proba(X_test),
          'classes':classes,
          'X_cols':X_cols,
          'y_coln':y_coln,
          'features':X_cols,
          'featimps':featimps,
          'y_pred':y_pred,
         'data_preds':data_preds}
    to_pkl(data,data_out_fh)            
    return grid_search,data_preds
        
def run_RF_regress(data_all,X_cols,y_coln,
                   test_size=0.5,data_test=None,data_out_fh=None):
    """
    This implements Random Forest classifier.
    
    :param data_all: dataframe with columns with features(Xs) and classes(y).
    :param X_cols: list of column names with features.
    :param y_coln: column name of column with classes.
    :param plot_fh: path to output plot file. 
    :returns grid_search: trained classifier object.
    :returns y_test: classes used for testing classifier. 
    :returns y_pred: predicted classes.
    :returns y_score: scores of predicted classes used to plot ROC curve.
    :returns feature_importances: relative importances of features (dataframe).
    """    
    from sklearn.ensemble import RandomForestRegressor

    X=data_all.loc[:,list(X_cols)]
    X=X.as_matrix()
    y=data_all.loc[:,y_coln]
    y=y.as_matrix()
    
    if test_size!=0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=88)
    else :
        X_train=X
        y_train=y
        X_test=data_test.loc[:,list(X_cols)].as_matrix()
        y_test=None

    model = RandomForestRegressor(random_state =88)
    param_grid = {"n_estimators": [3000],#[1000,2000,4000],#
                  "max_features": ['sqrt'],#[None,'sqrt','log2'],
                  "min_samples_leaf":  [1],#[1,25,50,100],
                  "criterion": ["mse"],
                  "oob_score": [True],
                 }

    grid_search = GridSearchCV(model, param_grid=param_grid,cv=10)
    grid_search.fit(X_train,y_train)
    y_pred=grid_search.predict(X_test)

    if test_size!=0:    
        data_preds=None
        # print grid_search.score(X_test, y_test)
    else:
        data_preds=data_test.loc[:,list(X_cols)]
        data_preds[y_coln]=y_pred

    featimps=pd.DataFrame(columns=['Feature','Importance'])
    featimps.loc[:,'Feature']=X_cols#[indices]
    featimps.loc[:,'Importance']=grid_search.best_estimator_.feature_importances_

    data={'RF_regress':grid_search,
          'X_train':X_train,
          'X_test':X_test,
          'y_train':y_train,
          'y_test':y_test,
          'X_cols':X_cols,
          'y_coln':y_coln,
          'features':X_cols,
          'featimps':featimps,
          'y_pred':y_pred,
         'data_preds':data_preds}
    to_pkl(data,data_out_fh)            
    return grid_search,data_preds    

def data_combo2ml(data_combo,data_fn,data_dh,plot_dh,
                ycoln,col_idx,
                ml_type='both',
                middle_percentile_skipped=0.1,
                force=False,
                ):
    """
    This runs the submodules to run classifier from fitness data (`data_combo`).
    
    :param basename(data_fn): in the form <data_combo>/<aas/cds>/<name of file>.
    :param data_feats: dataframe with features.
    :param y_coln: column name of column with classes (ys). 
    :param ml_type: classi | both
    """
    data_combo=del_Unnamed(data_combo)
    for dh in [plot_dh,data_dh]:
        if not exists(dh):
            makedirs(dh)
    # plot_cls_fh="%s/plot_ml_cls_%s.pdf" % (plot_dh,data_fn)
    # plot_reg_fh="%s/plot_ml_reg_%s.pdf" % (plot_dh,data_fn)
    data_combo_fh="%s/%s.input_raw" % (data_dh,data_fn)
    data_fh="%s/%s.cls.all" % (data_dh,data_fn)
    data_cls_train_fh="%s/%s.cls.train" % (data_dh,data_fn)
    data_cls_tests_fh="%s/%s.cls.tests" % (data_dh,data_fn)
    data_reg_train_fh="%s/%s.reg.train" % (data_dh,data_fn)
    data_reg_tests_fh="%s/%s.reg.tests" % (data_dh,data_fn)    
    pkld_cls_fh='%s/%s.cls.pkl' % (data_dh,data_fn)
    pkld_reg_fh='%s/%s.reg.pkl' % (data_dh,data_fn)
    # pkld_cls_metrics_fh='%s/%s.cls.metrics.pkl' % (data_dh,data_fn)
    pkld_reg_metrics_fh='%s/%s.reg.metrics.pkl' % (data_dh,data_fn)
    feature_importances_cls_fh="%s_%s_.csv" % (pkld_cls_fh,'featimps')

    y_coln_cls=ycoln
    y_coln_reg=ycoln

    if np.sum(~data_combo.loc[:,y_coln_cls].isnull())<50:
        logging.error("skipping %s: need more data: %d<50" %\
                        (data_fn,np.sum(~data_combo.loc[:,ycoln].isnull())))
        return False

    logging.info("processing: %s" % data_fn)
    if ml_type=='cls' or ml_type=='both':
        if not exists(pkld_cls_fh):
            if not exists(data_cls_train_fh):
                data_combo,data_ml,data_cls_train,data_cls_tests=make_cls_input(data_combo,
                                                                                y_coln_cls,
                                                                        middle_percentile_skipped=middle_percentile_skipped)
                data_combo.to_csv(data_combo_fh)
                data_ml.to_csv(data_fh)
                data_cls_train.to_csv(data_cls_train_fh)
                data_cls_tests.to_csv(data_cls_tests_fh)
            else:
                data_cls_train=pd.read_csv(data_cls_train_fh)
                data_cls_tests=pd.read_csv(data_cls_tests_fh)
                data_cls_train  =data_cls_train.set_index(col_idx,drop=True)
                data_cls_tests  =data_cls_tests.set_index(col_idx,drop=True)
            y_coln_cls="classes"
            logging.info("cls: train set = %d" % len(data_cls_train))
            X_cols_cls=data_cls_train.columns.tolist()
            X_cols_cls.remove(y_coln_cls)
            # cls
            pkld_cls,data_preds=run_RF_classi(data_cls_train,X_cols_cls,y_coln_cls,
                    test_size=0.34,data_out_fh=pkld_cls_fh) #                     
        else:
            logging.info('already exists: %s' % basename(pkld_cls_fh))
        if not exists(feature_importances_cls_fh):
	        get_RF_classi_metrics(pkld_cls_fh,data_dh=data_dh,plot_dh=plot_dh)
    
    if ml_type=='both':
        if not exists(pkld_reg_fh):
            if not exists('%s.train' % data_fh): 
                data_cls_tests=pd.read_csv(data_cls_train_fh)
                data_cls_train=pd.read_csv(data_cls_tests_fh)
                data_cls_tests  =data_cls_tests.set_index(col_idx,drop=True)
                data_cls_train  =data_cls_train.set_index(col_idx,drop=True)
                feature_importances_cls=pd.read_csv(feature_importances_cls_fh)
                data_reg_train,data_reg_tests=make_reg_input(data_combo,data_cls_train,data_cls_tests,
                                            feature_importances_cls,
                                            y_coln_reg,
                                            y_coln_cls="classes",
                                            topNfeats=25)       
                data_reg_train.to_csv(data_reg_train_fh)
                data_reg_tests.to_csv(data_reg_tests_fh)
            else:
                data_reg_train=pd.read_csv(data_cls_train_fh)
                data_reg_tests=pd.read_csv(data_cls_tests_fh)
                data_reg_train  =data_reg_train.set_index(col_idx,drop=True)
                data_reg_tests  =data_reg_tests.set_index(col_idx,drop=True)
            logging.info("reg: train set = %d" % len(data_reg_train))
            X_cols_reg=[c for c in data_reg_train.columns.tolist() if c!=y_coln_reg]
            # print data_reg_train.loc[:,X_cols_reg]
            pkld_reg_metrics,data_preds_reg_metrics=\
            run_RF_regress(data_reg_train,X_cols_reg,y_coln_reg,
                            test_size=0.34,data_out_fh=pkld_reg_metrics_fh)
            get_RF_regress_metrics(pkld_reg_metrics_fh,data_dh=data_dh,plot_dh=plot_dh)
        else:
            logging.info('already exists: %s' % basename(pkld_reg_fh))

def data_regress2data_fit(prj_dh,data_fit_key,
                          data_regress_all,col='FCA_norm'):
    # from dms2dfe.lib.io_nums import str2num
    from dms2dfe.lib.io_mut_files import rescale_fitnessbysynonymous,class_fit,mutids_converter

    data_fit=pd.read_csv("%s/%s" % (prj_dh,data_fit_key))
    data_fit=data_fit.loc[:,["mutids",col]].set_index("mutids",drop=True)
    data_fit_combo=data_fit.copy()
    data_fit_inferred=data_regress_all.reset_index().loc[:,["mutids",col]].set_index("mutids",drop=True)
    data_mutids_common=denanrows(data_fit.join(data_fit_inferred.loc[:,col],rsuffix='_inferred'))
    data_mutids_common=data_mutids_common.loc[(data_mutids_common.loc[:,data_mutids_common.columns[0]]!=data_mutids_common.loc[:,data_mutids_common.columns[1]]),:]

    for m in data_fit_combo.index.tolist():
        if pd.isnull(data_fit.loc[m,col]):
            if m in data_fit_inferred.index.tolist():
                data_fit_combo.loc[m,'inferred']=True
                data_fit_combo.loc[m,col]=data_fit_inferred.loc[m,col]
        else:
            data_fit_combo.loc[m,'inferred']=False
    for c in ['refi','ref','mut','refrefi']:
        data_fit_combo.loc[:,c]=mutids_converter(data_fit_combo.index.tolist(), c, 'aas')
    if col=='FCA_norm':
        data_fit_combo=rescale_fitnessbysynonymous(data_fit_combo,col_fit=col,col_fit_rescaled="FiA")
    data_fit_combo=class_fit(data_fit_combo)
    data_fit_combo.loc[:,'FiS']=\
    data_fit_combo.loc[(data_fit_combo.loc[:,'ref']==data_fit_combo.loc[:,'mut']),'FiA']
    data_fit_combo=data_fit_combo.sort_values(by="refi",axis=0)
    data_fit_combo.to_csv("%s/%s_inferred" % (prj_dh,data_fit_key))
    return data_fit_combo


#GB
from dms2dfe.lib.io_strs import get_time
from dms2dfe.lib.io_ml_data import feats_inter,keep_cols,feats_sel_corr,make_dXy,feats_inter_sel_corr
# %run ../../progs/dms2dfe/dms2dfe/lib/io_ml.py
# %run ../../progs/dms2dfe/dms2dfe/lib/io_ml_data.py
# %run ../../1_dms_software/progs/dms2dfe/dms2dfe/lib/io_ml_metrics.py

from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble.partial_dependence import plot_partial_dependence,partial_dependence
def run_est(est,X,y,params,cv=True):        
    if est=='GBR':
        est = GradientBoostingRegressor(random_state=88)
    elif est=='GBC':
        est = GradientBoostingClassifier(random_state=88)
    est.set_params(**params)
    if cv:
        r2s=cross_val_score(est,X,y,cv=10)
        print [r2s,np.mean(r2s)] 
    return r2s,est
def est2feats_imp(est,Xcols,Xy=None):
    try:
        feat_imp = pd.DataFrame(est.feature_importances_, Xcols)#.sort_values(ascending=False)
    except:
        est.fit(Xy[0],Xy[1])
        feat_imp = pd.DataFrame(est.feature_importances_, Xcols)#.sort_values(ascending=False)
    feat_imp.columns=['Feature importance']
    feat_imp=feat_imp.sort_values(by='Feature importance',ascending=False)
    return feat_imp

def dXy2ml(dXy,ycol,params=None,
            if_gridsearch=False,
            if_partial_dependence=False,
           if_feats_imps=False,
           inter=None,
           use_top=None,
           out_fh=None,
          regORcls='reg',
           force=False,cores=8):
    if out_fh is None:
        out_fh='%s_%s.pkl' % ('dXy2ml',get_time())

    if exists(out_fh) and (not force):
        try:
            dpkl=read_pkl(out_fh)
        except:
            return False
    else:
        dpkl={}

    if not ('dXy_final' in dpkl.keys()) or force:
        dpkl['dXy_input']=dXy
        dpkl['ycol']=ycol
        dXy_input=dXy.copy()
        to_pkl(dpkl,out_fh) #back        
        dXy,Xcols,ycol=make_dXy(dXy,ycol=ycol,
            if_rescalecols=True,
            unique_quantile=0.25)
        if len(dXy)<100:
            return False
        dpkl['dXy_preprocessed']=dXy
        to_pkl(dpkl,out_fh) #back

        dXy,Xcols,ycol=feats_sel_corr(dXy,ycol,range_coef=[0.9,0.8,0.7])
        dpkl['dXy_feats_sel_corr']=dXy
        to_pkl(dpkl,out_fh) #back

        dXy,Xcols,ycol=keep_cols(dXy,dXy_input,ycol)
        dpkl['dXy_feats_indi']=dXy
        to_pkl(dpkl,out_fh) #back

        if inter=='pre':
            dXy,Xcols,ycol=feats_inter_sel_corr(dXy,ycol,Xcols,dpkl['dXy_feats_indi'].copy(),
                                                top_cols=[
         'Conservation score (inverse shannon uncertainty): gaps ignored',#'Conservation score (ConSurf)',
         'Distance from active site residue: minimum',
         'Distance from dimer interface',
         'Temperature factor (flexibility)',
         'Residue depth'])
        
        dpkl['dXy_feats_inter_sel_corr']=dXy
        dpkl['dXy_final']=dXy
    else:
        dXy_input=dpkl['dXy_input']
        dXy=dpkl['dXy_final']
        ycol=dpkl['ycol']

    to_pkl(dpkl,out_fh) #back

    Xcols=[c for c in dXy.columns.tolist() if c!=ycol]
    X=dXy.loc[:,Xcols].as_matrix()
    y=dXy.loc[:,ycol].as_matrix()
    dpkl['X_final']=X
    dpkl['y_final']=y
    if regORcls=='reg':
        est_method='GBR'
    elif regORcls=='cls':
        est_method='GBC'

    if (if_gridsearch) or (params is None):
        if not ('gs_cv' in dpkl.keys()) or force:
            param_grid = {'learning_rate':[0.005,0.001,0.0001],#[0.1,0.01,0.005],# tuned with n estimators
                          'n_estimators':[1500,2000,3000,5000], # tuned with learning rate
                          'min_samples_leaf':[50,125], # lower -> less overfitting
                          'max_features':[None], 
                          'max_depth':[6],
                          'min_samples_split':[int(len(dXy)*0.05),int(len(dXy)*0.1),int(len(dXy)*0.25),int(len(dXy)*0.5)], # 0.5 to 1 of samples
                          'subsample':[0.8],
                          }
            if regORcls=='reg':
                param_grid['loss']=['ls', 'lad', 'huber']
                est_method='GBR'
                est = GradientBoostingRegressor(random_state=88)
            elif regORcls=='cls':
                param_grid['loss']=['deviance', 'exponential']
                est_method='GBC'
                est = GradientBoostingClassifier(random_state=88)
            logging.info('running grid search')
            gs_cv = GridSearchCV(est, param_grid, n_jobs=cores,cv=10).fit(X, y)
            print [gs_cv.best_params_,gs_cv.best_score_]
            params=gs_cv.best_params_
            dpkl['gs_cv']=gs_cv
            to_pkl(dpkl,out_fh) #back
            dpkl['params']=params
    
    if 'params' in dpkl.keys() and not force:
        params= dpkl['params']
    elif params is None:
        dpkl['params']=params
        
    if not ('est_all_feats_r2s' in dpkl.keys()) or force:
        r2s,est=run_est(est=est_method,X=X,y=y,params=params)
        dpkl['est_all_feats']=est
        dpkl['est_all_feats_r2s']=r2s

    if not ('feat_imp' in dpkl.keys()) or force:
        if if_gridsearch:
            feat_imp=est2feats_imp(dpkl['gs_cv'].best_estimator_,Xcols,Xy=None)
        else:
            feat_imp=est2feats_imp(est,Xcols,Xy=[X,y])
        dpkl['feat_imp']=feat_imp
        to_pkl(dpkl,out_fh) #back

    if if_feats_imps:
        fig=plt.figure(figsize=(5,10))
        ax=plt.subplot(111)
        feat_imp.plot(kind='barh', title='Feature Importances',ax=ax)
        ax.set_ylabel('Feature Importance Score')
        to_pkl(dpkl,out_fh) #back

    if not use_top is None:
        Xcols=dpkl['feat_imp'].head(use_top).index.tolist() #int(len(feat_imp)*0.15)
#         print Xcols[:use_top//5]
        if inter=='top':
            dXy,Xcols,ycol=feats_inter_sel_corr(dXy,ycol,Xcols,dXy_input,top_cols=Xcols[:len(Xcols)//5])
        X=dXy.loc[:,Xcols].as_matrix()
        y=dXy.loc[:,ycol].as_matrix()        
        r2s,est=run_est(est=est_method,X=X,y=y,params=params)
        feat_imp=est2feats_imp(est,Xcols,Xy=[X,y])
        dpkl['feat_imp_top_feats']=feat_imp
        dpkl['dXy_top_feats']=dXy
        dpkl['est_top_feats']=est
        dpkl['est_top_feats_r2s']=r2s
        to_pkl(dpkl,out_fh) #back
        
    if if_partial_dependence:
        feats_indi=[s for s in Xcols if not ((') ' in s) and (' (' in s))]
        features=[Xcols.index(f) for f in feats_indi]
        fig, axs = plot_partial_dependence(est, X, features,
                                           feature_names=Xcols,
                                           n_jobs=cores, grid_resolution=50,
                                          figsize=[10,30])
    to_pkl(dpkl,out_fh) #back
    # return est,dXy,dpkl

from dms2dfe.lib.io_ml_metrics import get_GB_cls_metrics

def data_fit2ml(dX_fh,dy_fh,info,regORcls='cls'):

    dy=pd.read_csv(dy_fh).set_index('mutids')
    dX=pd.read_csv(dX_fh).set_index('mutids')
    out_fh='%s/data_ml/%s.pkl' % (info.prj_dh,basename(dy_fh))
    if regORcls=='reg':
        ycol='FiA'
        dXy=pd.concat([dy.loc[:,ycol],dX],axis=1)
        dXy.index.name='mutids'
        params={'loss': 'ls', 'learning_rate': 0.001, 'min_samples_leaf': 50, 'n_estimators': 5000, 'subsample': 0.8, 'min_samples_split': 38, 'max_features': None, 'max_depth': 6}
    elif regORcls=='cls':
        ycol='class_fit_binary'
        dy.loc[(dy.loc[:,'class_fit']=='enriched'),ycol]=1
        dy.loc[(dy.loc[:,'class_fit']=='neutral'),ycol]=np.nan
        dy.loc[(dy.loc[:,'class_fit']=='depleted'),ycol]=0
        dXy=pd.concat([dy.loc[:,ycol],dX],axis=1)
        dXy.index.name='mutids'
    #     params={'loss': 'deviance', 'learning_rate': 0.0001, 'min_samples_leaf': 50, 'n_estimators': 3000, 'subsample': 0.8, 'min_samples_split': 23, 'max_features': None, 'max_depth': 6}
        params={'loss': 'exponential', 'learning_rate': 0.001, 'min_samples_leaf': 50, 'n_estimators': 1500, 'subsample': 0.8, 'min_samples_split': 23, 'max_features': None, 'max_depth': 6}
    dXy2ml(dXy,ycol,      
    #         params=params,
            if_gridsearch=True,
            if_partial_dependence=False,
    #        if_feats_imps=True,
            out_fh=out_fh,
            inter='pre',
            # force=True,
    #         use_top=25,
            regORcls=regORcls,
            cores=int(info.cores))
    
    # get metrics plots 
    get_GB_cls_metrics(data_fh=out_fh,info=info)