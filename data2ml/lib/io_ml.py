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

from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
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
from dms2dfe.lib.io_stats import get_regression_metrics
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

def data_fit2ml(data_fit_key,prj_dh,data_feats,
                data_fit_col='FCA_norm',data_fit_col_alt='FCA_norm',
                # data_fit_col='FiA',data_fit_col_alt='FiA',
                middle_percentile_skipped=0.1):
    """
    This runs the submodules to run classifier from fitness data (`data_fit`).
    
    :param data_fit_key: in the form <data_fit>/<aas/cds>/<name of file>.
    :param prj_dh: path to project directory.
    :param data_feats: dataframe with features.
    :param y_coln: column name of column with classes (ys). 
    """
    type_form='aas' # in mut_types_form:
    plot_dh="%s/plots/%s" % (prj_dh,type_form)
    data_dh="%s/data_ml/%s" % (prj_dh,type_form)
    plot_classi_fh="%s/fig_ml_classi_%s.pdf" % (plot_dh,data_fit_key.replace('/','_'))
    plot_regress_fh="%s/fig_ml_regress_%s.pdf" % (plot_dh,data_fit_key.replace('/','_'))
    data_fh="%s/data_ml_%s" % (data_dh,data_fit_key.replace('/','_'))
    grid_search_classi_fh=data_fh.replace("data_ml_","data_ml_classi_")+'.pkl'
    grid_search_regress_fh=data_fh.replace("data_ml_","data_ml_regress_")+'.pkl'
    grid_search_classi_metrics_fh=data_fh.replace("data_ml_","data_ml_classi_metrics_")+'.pkl'
    grid_search_regress_metrics_fh=data_fh.replace("data_ml_","data_ml_regress_metrics_")+'.pkl'

    # y_coln_classi="FCA_norm"
    y_coln_classi=data_fit_col
    if not exists(grid_search_regress_fh):
        data_fit_fh="%s/%s" % (prj_dh,data_fit_key)
        logging.info('processing: %s' % (basename(data_fit_fh)))
        data_fit=pd.read_csv(data_fit_fh)
        data_fit_col_muts=len(denanrows(data_fit.loc[:,data_fit_col]))
        data_fit_col_alt_muts=len(denanrows(data_fit.loc[:,data_fit_col_alt]))        
        if data_fit_col_muts<(0.8*data_fit_col_alt_muts):
            logging.warning("using %s instead of %s: bcz %s << %s" % (data_fit_col_alt,data_fit_col,data_fit_col_muts,data_fit_col_alt_muts))
            y_coln_classi=data_fit_col_alt
            data_fit_col=data_fit_col_alt            
        if np.sum(~data_fit.loc[:,y_coln_classi].isnull())>100:
            logging.info("processing: %s" % data_fit_key)
            if not exists(grid_search_classi_fh):
                if not exists(data_fh.replace("data_ml_","data_ml_classi_train_")):
                    # print sum(~pd.isnull(data_fit.loc[:,y_coln_classi]))
                    data_fit=y2classes(data_fit,y_coln_classi,
                                       middle_percentile_skipped=middle_percentile_skipped)
                    data_ml_mutids=list(data_fit.loc[:,'mutids'])                
                    y_coln_classi="classes"
                    # print sum(~pd.isnull(data_fit.loc[:,y_coln_classi]))
                    data_fit=set_index(data_fit,"mutids")
                    data_feats=set_index(data_feats,"mutids")
                    X_cols_classi=data_feats.columns.tolist()
                    data_combo=pd.concat([data_feats,
                                          data_fit.loc[:,y_coln_classi]],axis=1)
                    data_combo=del_Unnamed(data_combo)
                    # print sum(~pd.isnull(data_combo.loc[:,y_coln_classi]))
                    data_combo.index.name='mutids'
                    data_ml=X_cols2binary(data_combo.drop(y_coln_classi,axis=1))
                    data_ml.loc[:,y_coln_classi]=data_combo.loc[:,y_coln_classi]
                    data_ml=rescalecols(data_ml)
                    data_classi_train=denan(data_ml,axis='both',condi='all any')
                    # print sum(~pd.isnull(data_ml.loc[:,y_coln_classi]))

                    data_classi_train_mutids=list(data_classi_train.index.values)
                    data_classi_test_mutids=[mutid for mutid in data_ml_mutids if not mutid in data_classi_train_mutids]
                    data_classi_test=data_ml.loc[data_classi_test_mutids,:]

                    data_combo.reset_index().to_csv(data_fh.replace("data_ml_","data_ml_combo_"),index=False)
                    data_ml.reset_index().to_csv(data_fh.replace("data_ml_","data_ml_classi_all_"),index=False)
                    data_classi_train.reset_index().to_csv(data_fh.replace("data_ml_","data_ml_classi_train_"),index=False)
                    data_classi_test.reset_index().to_csv(data_fh.replace("data_ml_","data_ml_classi_test_"),index=False)
                else:
                    data_classi_train=pd.read_csv(data_fh.replace("data_ml_","data_ml_classi_train_"))
                    data_classi_test=pd.read_csv(data_fh.replace("data_ml_","data_ml_classi_test_"))

                    data_classi_train  =data_classi_train.set_index("mutids",drop=True)
                    data_classi_test  =data_classi_test.set_index("mutids",drop=True)
                    y_coln_classi="classes"
                logging.info("number of mutants used for training = %d" % len(data_classi_train))
                # logging.info("this step would take a 10 to 15min to complete.")

                X_cols_classi=data_classi_train.columns.tolist()
                X_cols_classi.remove(y_coln_classi)
                # classi
                grid_search_classi,data_preds=run_RF_classi(data_classi_train,X_cols_classi,y_coln_classi,
                        test_size=0.34,data_test=data_classi_test,data_out_fh=grid_search_classi_fh) #                     
                get_RF_classi_metrics(grid_search_classi_fh,data_dh='data_ml/',plot_dh='plots/')
                
                # # classi metrics
                # feature_importances_classi_fh="%s_%s_.csv" % (grid_search_classi_fh,'featimps')
                # feature_importances_classi=pd.read_csv(feature_importances_classi_fh)
                # X_cols_classi_metrics_selected=feature_importances_classi\
                # .sort_values(by='Importance',ascending=False).head(25).loc[:,'Feature'].tolist()
                # X_cols_classi_metrics=[col for col in X_cols_classi if col in X_cols_classi_metrics_selected]            
                # grid_search_classi,data_preds=run_RF_classi(data_classi_train,X_cols_classi_metrics,y_coln_classi,
                #         test_size=0.34,data_test=data_classi_test,data_out_fh=grid_search_classi_metrics_fh) 
                # get_RF_classi_metrics(grid_search_classi_metrics_fh,data_dh='data_ml/',plot_dh='plots/')
                
            y_coln_classi="classes"
            feature_importances_classi_fh="%s_%s_.csv" % (grid_search_classi_fh,'featimps')
            feature_importances_classi=pd.read_csv(feature_importances_classi_fh)
            data_classi_test=pd.read_csv(data_fh.replace("data_ml_","data_ml_classi_test_"))
            data_classi_test  =data_classi_test.set_index("mutids",drop=True)
            data_classi_train=pd.read_csv(data_fh.replace("data_ml_","data_ml_classi_train_"))
            data_classi_train  =data_classi_train.set_index("mutids",drop=True)
            #regress
            y_coln_regress=data_fit_col
            data_regress_train=data_classi_train                
            data_regress_train=X_cols2binary(data_regress_train,[y_coln_classi])
            data_fit=set_index(data_fit,"mutids")
            data_regress_train.loc[:,y_coln_regress]=\
            data_fit.loc[data_classi_train.index.values,y_coln_regress]
            data_regress_train=denan(data_regress_train,axis='both',condi='all any')
            data_regress_test=data_classi_test                
            if y_coln_classi in data_regress_test.columns.tolist(): 
                data_regress_test=data_regress_test.drop(y_coln_classi,axis=1)
            else:
                data_regress_test=denan(data_regress_test,axis='both',condi='all any')
                data_regress_test=X_cols2binary(data_regress_test,[y_coln_classi])
            data_regress_test=denan(data_regress_test,axis='both',condi='all any')
            X_cols_regress_class_fit=[]
            X_cols_regress_selected=feature_importances_classi\
            .sort_values(by='Importance',ascending=False).head(25).loc[:,'Feature'].tolist()
            X_cols_regress=[col for col in X_cols_regress_class_fit+X_cols_regress_selected\
                            if col in data_regress_test.columns.tolist()]
            if y_coln_regress in X_cols_regress:
                X_cols_regress.remove(y_coln_regress)
            data_regress_train.to_csv(data_fh.replace("data_ml_","data_ml_regress_train_"))
            data_regress_test.to_csv(data_fh.replace("data_ml_","data_ml_regress_test_"))
            # try:
            grid_search_regress_metrics,data_preds_regress_metrics=\
            run_RF_regress(data_regress_train,X_cols_regress,y_coln_regress,
                            test_size=0.34,data_test=data_regress_test,data_out_fh=grid_search_regress_metrics_fh)
            get_RF_regress_metrics(grid_search_regress_metrics_fh,data_dh='data_ml/',plot_dh='plots/')

            logging.info("number of mutants used for training = %d" % len(data_regress_train))
            grid_search_regress,data_preds_regress=\
            run_RF_regress(data_regress_train,X_cols_regress,y_coln_regress,
                            test_size=0,data_test=data_regress_test,data_out_fh=grid_search_regress_fh)
            data_preds_regress.to_csv(data_fh.replace("data_ml_","data_ml_regress_preds_"))
            data_regress_all=data_preds_regress.append(data_regress_train)
            data_regress_all.to_csv(data_fh.replace("data_ml_","data_ml_regress_all_"))
            data_regress2data_fit(prj_dh,data_fit_key,data_regress_all,col=y_coln_regress)
            return data_regress_all
            # except:
                # logging.info("skipping: %s : requires more data" % basename(data_fit_key))
        else:
            logging.info("skipping %s: requires more samples %d<10" %\
                            (data_fit_key,np.sum(~data_fit.loc[:,y_coln_classi].isnull())))
    else:
        data_regress_all=pd.read_csv(data_fh.replace("data_ml_","data_ml_regress_all_"))
        data_regress2data_fit(prj_dh,data_fit_key,data_regress_all)
        return data_regress_all
    
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
                data_combo,data_ml,data_cls_train,data_cls_tests=make_cls_input(data_combo,y_coln_cls,                                                                                middle_percentile_skipped=middle_percentile_skipped)
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
    
    
# from dms2dfe.lib.io_nums import is_numeric
# from dms2dfe.ana4_modeller import get_cols_del
# from dms2dfe.lib.io_dfs import set_index

def get_abs_diff(mutid_mm,data_feats,col):
    mutid_m1,mutid_m2=mutid_mm.split(':')
    return abs(data_feats.loc[mutid_m1,col]-data_feats.loc[mutid_m2,col])
def add_feats(data_feats_all_mm,data_feats_all):
    for col in data_feats_all:
#         print col
        data_feats_all_mm.loc[:,"$\Delta$(%s)" % col]=data_feats_all_mm.apply(lambda x: \
                                    get_abs_diff(x['mutids'],data_feats_all,col), axis=1)
    return data_feats_all_mm

def make_data_combo(data_fit_dm,data_feats,ycol,Xcols):
    cols_numeric=[c for c in data_feats if is_numeric(data_feats.loc[:,c])]
    data_feats=data_feats.loc[:,cols_numeric]
    for col in get_cols_del(data_feats):
        del data_feats[col]
    data_feats_dm=pd.DataFrame(index=data_fit_dm.index)
    data_feats_dm.loc[:,ycol]=data_fit_dm.loc[:,ycol]
    data_feats_dm=add_feats(data_feats_dm.reset_index(),data_feats)
    return data_feats_dm
    
def get_cols_del(data_feats):
    cols_del=[col for col in data_feats if "Helix formation" in col]+\
    [col for col in data_feats if "beta bridge" in col]+\
    [col for col in data_feats if "Chirality" in col]+\
    [col for col in data_feats if "Offset from residue to the partner" in col]+\
    [col for col in data_feats if "Energy (kcal/mol) of " in col]+\
    [col for col in data_feats if "Secondary structure" in col]+\
    [col for col in data_feats if 'cosine of the angle between C=O of residue and C=O of previous residue' in col]+\
    [col for col in data_feats if '$\Delta$(Molecular Polarizability) per substitution' in col]+\
    [col for col in data_feats if '$\Delta$(Molecular weight (Da)) per substitution' in col]+\
    [col for col in data_feats if '$\Delta$(Molecular Refractivity) per substitution' in col]+\
    [col for col in data_feats if 'bend' in col]
    return cols_del
