#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_ml_metrics``
================================
"""
from os.path import abspath,dirname,exists,basename
from os import makedirs

from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,regression

from data2ml.lib.io_data_files import read_pkl,to_pkl
from data2ml.lib.io_dfs import set_index,denan,denanrows,del_Unnamed
from data2ml.lib.io_nums import is_numeric
from data2ml.lib.io_strs import linebreaker
from data2ml.lib.io_plots import saveplot,get_axlims

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # no Xwindows

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
from data2ml.lib.io_strs import get_logger
logging=get_logger()
# logging.basicConfig(format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s',level=logging.DEBUG) # filename=cfg_xls_fh+'.log'

def plot_ROC(y_test,y_score,classes,lw=2,
             ax_roc=None,annotate=True,
            get_auc=False,
            reference_line=True,plot_fh=None):
    """
    This plots ROC curve.
    
    :param y_test: test split used for predictions.
    :param y_score: probabilities of predictions.
    :param classes: list with unique classes in y
    """
    # fig = 
    if ax_roc is None: 
        plt.figure(figsize=(3,3),dpi=300)#figsize=(11,5))
        ax_roc = plt.subplot(111)
        
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    classes_to_plot=[]
    if len(classes)>2:
        for classi in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test[:, classi], y_score[0][:, 1])
            if auc(fpr,tpr)<0.5:
                fpr, tpr, _ = roc_curve(y_test[:, classi], y_score[0][:, 0])
            if len(np.unique(tpr))>10:
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0

                ax_roc.plot(fpr, tpr,lw=lw)#, label="%s (AUC=%.2f)" % (classes[classi],auc(fpr,tpr)))
                ax_roc.annotate("%s (AUC=%.2f)" % (classes[classi],auc(fpr,tpr)), xy=(2, 1), xytext=(2, 1))
                classes_to_plot.append(classi)
        if len(classes_to_plot)!=0:
            mean_tpr /= len(classes_to_plot)
            mean_tpr[-1] = 1.0
            ax_roc.plot(mean_fpr, mean_tpr, lw=lw)#, label="%s (AUC=%.2f)" % ("mean",auc(mean_fpr,mean_tpr)))
            logging.info("mean AUC = %.2f" % auc(fpr,tpr))
    else:
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        if auc(fpr,tpr)<0.5:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 0])
            logging.info("mean AUC = %.2f" % (1-auc(fpr,tpr)))
        else:
            logging.info("mean AUC = %.2f" % auc(fpr,tpr))
        ax_roc.plot(fpr, tpr,lw=lw)#, label="%s (AUC=%.2f)" % ("mean",auc(fpr,tpr)))

    # get auc score
    if auc(fpr,tpr)<0.5:
        auc_score=1-auc(fpr,tpr)
    else:
        auc_score=auc(fpr,tpr)
    if annotate:
        ax_roc.annotate("AUC = %.2f" % auc_score, 
                    xy=(0.45, 0), xytext=(0.45, 0))
    if reference_line:
        ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlim([-0.01,1.01])
    ax_roc.set_ylim([-0.01,1.01])
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
#     ax_roc.legend(loc='lower right',
#                  )
#     ax_roc.grid(color='lightgrey')
#     plt.axis('equal')
    # plt.tight_layout()
    saveplot(plot_fh,transparent=False)
    if get_auc:
        return auc_score
    else:
        return ax_roc       
def plot_importances(feature_importances,plot_fh=None,data_out_fh=None):
    """
    This plots relative importances of features.
    
    :param importances: relative importances of features from `.best_estimator_.feature_importances_`
    :param X_cols: list of features in same order as `importances`
    :returns feature_importances: dataframe with relative importances of features.
    """
    fig = plt.figure(figsize=(8,len(feature_importances)*0.25))#figsize=(11,5))
    ax_imp = plt.subplot(1,1,1)

    feature_importances=feature_importances.sort_values("Importance",axis=0,ascending=False)
    feature_importances.plot(kind="barh",ax=ax_imp,legend=False)
    ax_imp.invert_yaxis()
    ax_imp.set_yticklabels(feature_importances.loc[:,"Feature"])
    ax_imp.set_xlabel("Feature importance")
    ax_imp.set_xticklabels(ax_imp.get_xticks(),rotation=90)
    ax_imp.grid()
    saveplot(plot_fh)
    if not data_out_fh is None:
        feature_importances.to_csv(data_out_fh)
    return feature_importances

import forestci as fci
from data2ml.lib.io_stats import get_regression_metrics
def get_RF_ci(RF_type,RF_classi,X_train,X_test,y_test,y_score,
                classes=['yes','no'],plot_fh=None):
    # calculate inbag and unbiased variance
    inbag = fci.calc_inbag(X_train.shape[0], RF_classi)
    V_IJ_unbiased = fci.random_forest_error(RF_classi,inbag, X_train,
                                                 X_test)
    # Plot forest prediction for emails and standard deviation for estimates
    # Blue points are spam emails; Green points are non-spam emails
    idx = np.where(y_test == 1)[0]
    fig=plt.figure(figsize=[3,3])
    ax=plt.subplot(111)
    if RF_type=='classi':
        ax.errorbar(y_score[idx, 1], np.sqrt(V_IJ_unbiased[idx]),
                     fmt='.', alpha=0.75, label=classes[0])

        idx = np.where(y_test == 0)[0]
        ax.errorbar(y_score[idx, 1], np.sqrt(V_IJ_unbiased[idx]),
                     fmt='.', alpha=0.75, label=classes[1])

        ax.set_xlabel('Prediction probability')
        ax.set_ylabel('Standard deviation')
        space=0.3
        ax.set_ylim([ax.get_ylim()[0]*(1+space),
                     ax.get_ylim()[1]*(1+space)])
        leg=ax.legend(loc='upper right',frameon=True)
        leg.get_frame().set_alpha(0.5)
        # plt.axis('equal')
    if RF_type=='regress':
        # Plot error bars for predicted MPG using unbiased variance
        ax.errorbar(y_test, y_score, yerr=np.sqrt(V_IJ_unbiased), fmt='o')
        xlim,ylim=get_axlims(y_test,y_score,
                             space=0.1,equal=True)
        ax.plot(xlim,xlim, '--',color='gray')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Test')
        ax.set_ylabel('Predicted')
        results,_,_=get_regression_metrics(y_test,y_score)
        logging.info(results.replace('\n',' '))
        ax.text(0, 1, results,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
        data_regress=pd.DataFrame({'y_test':y_test,
                                    'y_pred':y_score,
                                    'err':np.sqrt(V_IJ_unbiased)
                                    })
        if not plot_fh is None:
            data_regress.to_csv('%s.csv' % plot_fh)
    ax.grid(True)
    saveplot(plot_fh)
    
def get_RF_cm(y_test, y_pred,classes,plot_fh=None,data_out_fh=None):
    fig=plt.figure(figsize=[2.5,2])
    data=pd.DataFrame(confusion_matrix(y_test, y_pred))
    data.columns=classes
    data.index=classes
    ax=sns.heatmap(data,cmap='bone_r')
#     plt.axis('equal')
    saveplot(plot_fh)
    if not data_out_fh is None:
        data.to_csv(data_out_fh)

def get_RF_cr(y_test,y_pred,classes,data_out_fh=None):
    s=classification_report(y_test, y_pred)
    for i,line in enumerate(s.split('\n')):
        line=line.replace(' / ','/')
        if not line=='':
            if i==0:
                cols=line.split()
                data=pd.DataFrame(columns=cols)
            else:
                for coli,col in enumerate(cols):
                    data.loc[line.split()[0],cols[coli]]=line.split()[coli+1]            

    data.index=list(classes)+[data.index.tolist()[2]]
    if not data_out_fh is None:
        data.to_csv(data_out_fh)
    return data

def get_RF_regress_metrics(data_regress_fh,data_dh='data_ml/',plot_dh='plots/'):
    data_regress=read_pkl(data_regress_fh)
    RF_regress=data_regress['RF_regress']
    X_train=data_regress['X_train']
    X_test=data_regress['X_test']
    y_test=data_regress['y_test']
    y_pred=data_regress['y_pred']
    featimps=data_regress['featimps']
    #featimps
    plot_type='featimps'
    plot_fh="%s_%s_.pdf" % (data_regress_fh.replace(data_dh,plot_dh),plot_type)
    data_out_fh="%s_%s_.csv" % (data_regress_fh,plot_type)
    importances = RF_regress.best_estimator_.feature_importances_
    feature_importances=plot_importances(featimps,plot_fh=plot_fh,data_out_fh=data_out_fh)

    # ci :confidence intervals
    plot_type='ci'
    plot_fh="%s_%s_.pdf" % (data_regress_fh.replace(data_dh,plot_dh),plot_type)
    get_RF_ci('regress',gcv2rfc(RF_regress),X_train,X_test,
                 y_test,y_pred,plot_fh=plot_fh)    
    
def gcv2rfc(gridcv):
    classi=gridcv.best_estimator_
    classi.n_estimators=gridcv.param_grid['n_estimators'][0]
    classi.estimators_=gridcv.best_estimator_.estimators_
    return classi

def get_RF_classi_metrics(data_classi_fh,data_dh='data_ml/',plot_dh='plots/'):
    data_classi=read_pkl(data_classi_fh)
    RF_classi=data_classi['RF_classi']
    X_train=data_classi['X_train']
    X_test=data_classi['X_test']
    y_test=data_classi['y_test']
    y_pred=data_classi['y_pred']
    y_score=data_classi['y_score']
    classes=data_classi['classes']
    featimps=data_classi['featimps']
    #roc
    plot_type='roc'
    plot_fh="%s_%s_.pdf" % (data_classi_fh.replace(data_dh,plot_dh),plot_type)
    plot_ROC(y_test,y_score,classes,plot_fh=plot_fh)
    #featimps
    plot_type='featimps'
    plot_fh="%s_%s_.pdf" % (data_classi_fh.replace(data_dh,plot_dh),plot_type)
    data_out_fh="%s_%s_.csv" % (data_classi_fh,plot_type)
    plot_importances(featimps,plot_fh=plot_fh,data_out_fh=data_out_fh)

    # ci :confidence intervals
    plot_type='ci'
    plot_fh="%s_%s_.pdf" % (data_classi_fh.replace(data_dh,plot_dh),plot_type)
    get_RF_ci('classi',gcv2rfc(RF_classi),X_train,X_test,
                 y_test,y_score,classes=classes,plot_fh=plot_fh)    

    # cm : confusion matrix
    plot_type='cm'
    plot_fh="%s_%s_.pdf" % (data_classi_fh.replace(data_dh,plot_dh),plot_type)
    data_out_fh="%s_%s_.csv" % (data_classi_fh,plot_type)
    get_RF_cm(y_test, y_pred,classes,plot_fh=plot_fh,data_out_fh=data_out_fh)
    #cr : classi report
    plot_type='cr'
    plot_fh="%s_%s_.pdf" % (data_classi_fh.replace(data_dh,plot_dh),plot_type)
    data_out_fh="%s_%s_.csv" % (data_classi_fh,plot_type)
    get_RF_cr(y_test,y_pred,classes,data_out_fh=data_out_fh)

from data2ml.lib.io_ml import run_est
def get_GB_cls_metrics(data_fh,cores=2,out_dh=None,force=False):
    from pylab import figtext
    try:
        dpkl=read_pkl(data_fh)
    except:
        return False
    if not 'gs_cv' in dpkl.keys():
        return False
    if out_dh is None:
        out_dh=dirname(data_fh)
    dXy=dpkl['dXy_final']
    ycol=dpkl['ycol']
    gs_cv=dpkl['gs_cv']
    feat_imp = dpkl['feat_imp']

    Xcols=[c for c in dXy.columns.tolist() if c!=ycol]
    est=gs_cv.best_estimator_
    X=dXy.loc[:,Xcols].as_matrix()
    y=dXy.loc[:,ycol].as_matrix()        

    #partial dep 
    plot_type='partial_dep'
    plot_fh='%s.%s.pdf' % (basename(data_fh),plot_type)
    logging.info('ml plots: %s' % plot_fh)
    if (not exists(plot_fh)) or force:

        # feats_indi=[s for s in Xcols if not ((') ' in s) and (' (' in s))]
        # features=[Xcols.index(f) for f in feats_indi]
        if dpkl['if_small_data']==True:
            logging.warning("getting partial_dependence using no params; bcz of small data(?)")
            _,est=run_est('GBC',None,None,
                            params={},
                            cv=False)
            est.fit(dpkl['X_final'],dpkl['y_final'])
        # fig, axs = plot_partial_dependence(est, dpkl['X_final'], features,
        #                                    feature_names=Xcols,
        #                                    n_jobs=cores, grid_resolution=50,
        #                                   figsize=[10,30])

        feats_indi=[s for s in dpkl['feat_imp'].sort_values(by='Feature importance',ascending=True).head(6).index.tolist() if not ((') ' in s) and (' (' in s))]
        print feats_indi
        features=[Xcols.index(f) for f in feats_indi]
        feature_names=linebreaker(Xcols)
        from sklearn.ensemble.partial_dependence import plot_partial_dependence

        fig, axs = plot_partial_dependence(est, dpkl['X_final'], features,#[[features[1],features[2]]],
                                           feature_names=feature_names,
                                           n_jobs=int(cores), grid_resolution=50,
                                           n_cols=6,
                                           line_kw={'color':'r'},
                                          figsize=[18,3])
        figtext(0.9,-0.2,'AUC = %.2f' % gs_cv.best_score_,ha='right',color='b')
        saveplot(plot_fh,form='pdf',tight_layout=False)
    
    #relimp
    plot_type='featimps'
    plot_fh='%s.%s.pdf' % (basename(data_fh),plot_type)
    if (not exists(plot_fh)) or force:
        featst=10
        fig=plt.figure(figsize=(3,featst*0.75))
        # fig = plt.figure(figsize=(8,featst*0.25))#figsize=(11,5))
        ax=plt.subplot(111)
        feat_imp=feat_imp.sort_values(by='Feature importance',ascending=True)
        feat_imp.index=linebreaker(feat_imp.index, break_pt=30)
        feat_imp.tail(featst).plot(kind='barh',ax=ax, color='red')
        ax.set_xlabel('Feature Importance')
        ax.legend([])    
        figtext(0.9,-0.2,'AUC = %.2f' % gs_cv.best_score_,ha='right',color='b')
        saveplot(plot_fh,form='pdf',tight_layout=False)
