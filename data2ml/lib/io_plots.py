#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.use('Agg') # no Xwindows
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
import numpy as np

from data2ml.lib.io_dfs import debad,denanrows
# import seaborn as sns

def get_rgb_colors(series,cmap='Greens',res=10):
    series=(series-series.min())/(series.max()-series.min())*10
    series=series.fillna(0)
    series=series.astype(int)
    # print series
    try:
        series=[i[0] for i in series.values]
    except:
        series=series.tolist()
    # import seaborn as sns
    color_palette=sns.color_palette(cmap, res)
    colors=[]
    for i in series:
        for ci in range(1,len(color_palette)+1):
            if i==ci:
                colors.append(color_palette[ci-1])
                break
            elif i==0: #nans
                colors.append((1,1,1))
                break
    if len(series)==len(colors):
        return colors


def saveplot(plot_fh,form='both',
            transparent=False,
            tight_layout=True,
            print_fh=False):
    if not plot_fh is None:
        def save(plot_fh,form,transparent):
            if '.%s' % form in plot_fh:
                plot_out_fh=plot_fh
            else:
                plot_out_fh='%s.%s' % (plot_fh,form)
            if print_fh:
                print plot_out_fh
            plt.savefig(plot_out_fh,format=form,
                transparent=transparent,
                dpi=300)
        if tight_layout:
            plt.tight_layout()
        if plot_fh!=None:
            if form=="pdf" or form=="both":
                save(plot_fh,'pdf',transparent)
            if form=="png" or form=="both":
                save(plot_fh,'png',transparent)
        plt.clf()
        plt.close()

def get_axlims(X,Y,space=0.2,equal=False):
    xmin=np.min(X)
    xmax=np.max(X)
    xlen=xmax-xmin
    ymin=np.min(Y)
    ymax=np.max(Y)
    ylen=ymax-ymin
    xlim=(xmin-space*xlen,xmax+space*xlen)
    ylim=(ymin-space*ylen,ymax+space*ylen)
    if not equal:
        return xlim,ylim
    else:
        lim=[np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]])]
        return lim,lim

def repel_labels(ax, x, y, labels, k=0.01,
                 arrow_color=None,label_color='k',fontsize=7,
                 ha='right',va='center'):
    if arrow_color is None:
        arrow_color='royalblue'
    
    import networkx as nx
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    color=label_color,
                    fontsize=fontsize,
                    ha=ha,
                    va=va,
#                     weight="bold",
                    arrowprops=dict(arrowstyle="-",
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3", 
                                    color=arrow_color), )
    # expand limits
    all_pos = np.vstack(pos.values())
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos-x_span*0.15, 0)
    maxs = np.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])
    return ax

# from scipy import stats
from data2ml.lib.io_stats import get_regression_metrics
def plot_scatter_reg(data_all,cols,
                     xlabel=None,ylabel=None,title=None,
                     color_scatter="gray",
                     ms_scatter=20,
                     color_line='k',
                     logscale=False,
                     std=False,
                     regline=True,
                     results=True,
                     results_n=False,
                     results_RMSE=True,
                     result_spearman=False,
                     space=0.2,figsize=[2,2],
                     ax=None,plot_fh=None):
    if ax==None:
        fig=plt.figure(figsize=figsize,dpi=300)
        ax=plt.subplot(111)
    data_all=debad(data_all.loc[:,cols],axis=0,condi='any',bad='nan')
    if logscale:
        data_all=debad(data_all,axis=0,condi='any',bad=0)
        data_all=data_all.apply(np.log2)
    if regline:
        ax=sns.regplot(data=data_all,x=cols[0],y=cols[1],
                    line_kws={"color":color_line},
                    scatter_kws={"color":color_scatter,
                                's':ms_scatter},
                    ax=ax)
    else:
        data_all.plot.scatter(x=cols[0],y=cols[1],
                              color='b',
                              alpha=0.3,
                              ax=ax)
    # if std:
    #     ax.scatter(data_all.loc[:,cols[0]],data_all.loc[:,cols[1]],
    #                # s=ms, 
    #                c=data_all.loc[:,group_col],cmap=cmap,edgecolor='none',
    #                       alpha=scatter_groups_alpha,zorder=zorder)

    # r, _ = stats.pearsonr(data_all.loc[:,cols[0]],
    #                      data_all.loc[:,cols[1]])
    results,_,_=get_regression_metrics(data_all.loc[:,cols[0]],data_all.loc[:,cols[1]])
    if results_n:
        results='%s\nn=%s' % (results,len(denanrows(data_all.loc[:,cols])))
    if not results_RMSE:
        results=results.split('\n')[0]
    if result_spearman:
        results='%s = %.2f' % (r'$\rho$',data_all.loc[:,cols].corr(method='spearman').iloc[0,:][1])
    if results:
        ax.text(0, 1, results,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    # ax.legend(["r=%.2f\nRMSE" % (r)],loc="upper left")
    if not title is None:
        # sns.plt.title(title)
        ax.set_title(title)
    xlim,ylim=get_axlims(data_all.loc[:,cols[0]],
                 data_all.loc[:,cols[1]],space=space)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    saveplot(plot_fh,form='both',transparent=False)
    return ax

def heatmap(data,label_suffix=None,
                      clim=[None,None],cmap='winter',
                      rescale=None,
                      logscale=False,
                      sortby=None,
                      sortascending=True,  
                      threshold=None, 
                      ylabel=None,label_cbar='',
                      title=None,
                    yticklabels_linebreak=False,
                      yticklabels_linebreak_pt=16,
                      figsize=None,plot_fh=None):
    if yticklabels_linebreak:
        data.index=linebreaker(data.index,
                           break_pt=yticklabels_linebreak_pt)
    # data.drop(['Selection','All'],axis=1)
    if figsize is None:
        figsize=[data.shape[1]*0.7,data.shape[0]*0.7]
        # print figsize
        fig=plt.figure(figsize=figsize,dpi=300)
    elif len(figsize)==2:
        fig=plt.figure(figsize=figsize,dpi=300)
    if not sortby is None:
        data=data.sort_values(by=sortby,ascending=sortascending)
    else:
        if not label_suffix is None:
            if label_suffix in data: 
                if len(data.index)>0:
                    yticklabels_tups=zip(data.index,data.loc[:,label_suffix])
                else:
                    yticklabels_tups=zip(data.index,data.loc[:,label_suffix])
                yticklabels=[]
            #     print yticklabels_tups
                for tup in yticklabels_tups:
            #         print tup[1] 
                    if (pd.isnull(tup[1])) or (tup[1]==np.nan) or (tup[1]=='nan'):
                        yticklabels.append("%s\n%s %s" % (tup[0],tup[1],label_suffix))    
                    else:
                        yticklabels.append("%s\n%s %s" % (tup[0],tup[1],label_suffix))    
                del data[label_suffix]

        # else:
        #     yticklabels=data.index.tolist()
        # print data
        # data.index=yticklabels
    if not threshold is None:
        for col in data: 
            data.loc[data.loc[:,col]>threshold,col]=np.nan
    if logscale:
        data=data.astype('float64').apply(np.log10)
        data_nan=data.replace(-np.inf,np.nan)
        data=data.replace(-np.inf, data_nan.min().T.min())
    if rescale=='rows':
        data=rescale_df(data.T).T
    elif rescale=='cols':
        data=rescale_df(data)    
    # print data
    import seaborn as sns
    if clim[0] is None:
        ax=sns.heatmap(data,cmap=cmap)
    else:
        ax=sns.heatmap(data,cmap=cmap,vmin=clim[0],vmax=clim[1])

    ax.xaxis.set_ticks_position('top')
    if ylabel!=None:
        loc=[-1,data.shape[0]+0.1]
        ax.text(loc[0],loc[1],ylabel,
               va='bottom',ha='center')
#         ax.set_ylabel(ylabel) 
    if title!=None:
        ax.set_title(title) 
    if label_cbar!=None:
        loc=[data.shape[1],-len(data)/15]
        ax.text(loc[0],loc[1],label_cbar,
               va='center',ha='left')
#         ax.annotate(label_cbar,xy=loc,xytext=loc,zorder=1)
    if plot_fh!=None:
        plt.savefig(plot_fh,format='pdf',
#                     transparent=True
                   )
    return data

def annot_corners(labels,X,Y,ax,space=-0.2,fontsize=18):
    xlims,ylims=get_axlims(X,Y,space=space)
    
    labeli=0
    for x in xlims:
        for y in ylims:
            ax.text(x,y,labels[labeli],
                color='k',
                fontsize=fontsize,
                ha='center',
                va='center',
                bbox=dict(facecolor='w',edgecolor='none',alpha=0.4),
                )
            labeli+=1
    return ax

def plot_contourf(x,y,z,contourlevels=15,xlabel=None,ylabel=None,
                scatter=False,contour=False,
                annot_fit_land=True,
                cmap="coolwarm",cbar=True,cbar_label="",
                a=0.5,vmin=None,vmax=None,interp='linear',#'nn',
                xlog=False,
                fig=None,ax=None,plot_fh=None):
    from matplotlib.mlab import griddata
    xi=get_linspace(x)
    yi=get_linspace(y)
    zi = griddata(x, y, z, xi, yi, interp=interp)#interp) #interp)
    
    if fig==None:
        fig=plt.figure(figsize=[6,3])
    if ax==None: 
        ax=plt.subplot(121)
    #contour the gridded data, plotting dots at the nonuniform data points.
    
    if vmax==None: 
        vmax=abs(zi).max()
    if vmin==None: 
        vmin=abs(zi).min()
    #print vmin
    if contour:
        CS = ax.contour(xi, yi, zi, contourlevels, linewidths=0.5, colors='k',alpha=a)
    CS = ax.contourf(xi, yi, zi, contourlevels, 
                      cmap=cmap,
                      vmax=vmax, vmin=vmin)
    if cbar:
        # colorbar_ax = fig.add_axes([0.55, 0.15, 0.035, 0.5]) #[left, bottom, width, height]
        colorbar_ax = fig.add_axes([0.55, 0.3, 0.035, 0.5]) #[left, bottom, width, height]
        colorbar_ax2=fig.colorbar(CS, cax=colorbar_ax,extend='both')
        colorbar_ax2.set_label(cbar_label)
        clim=[-4,4]
        colorbar_ax2.set_clim(clim[0],clim[1])
    # plot data points.
    if scatter:
        ax.scatter(x, y, marker='o', c='b', s=5, zorder=10)

    if annot_fit_land:
        labels=["$F:cB$","$F:B$","$cF:cB$","$cF:B$"]
        # labels=["$F,B$","$F,cB$","$cF,B$","$cF,cB$"]
        if xlog:
            # x=np.log2(x)+1
            # x=x-1.5
            x=x/1.61
        ax=annot_corners(labels,x,y,ax,fontsize=15)
    if plot_fh!=None:
        fig.savefig(plot_fh,format="pdf")
    return ax


def get_linspace(x,n=100):
    return np.linspace(np.min(x),np.max(x),n)
def get_grid(x,y,n=100):
#     return np.linspace(np.min(x),np.max(x),n)
    return np.mgrid[np.min(x):np.max(x), np.min(y):n:np.max(y)]

def scatter_colorby_groups(data,x_col,y_col,group_col,contour_col,
                           scatter_groups=True,contourf=False,
                           contourlevels=15,
                           contourinterp='nn',
                           scatter_groups_order=None,scatter_groups_cmap='YlGn',
                           ms=10,
                           cbar=True,cbar_label="",cbar_min=None,cbar_max=None,
                           cmap="Blues",scatter_groups_alpha=0.8,
                           scatter_groups_legend=True,legend_loc='out',
                           xlog=False,
                           fig=None,ax=None,zorder=2):
    if ax==None:
        fig=plt.figure(figsize=[3,3],dpi=300)
        ax=plt.subplot(111)
    if scatter_groups:
        if not is_numeric(data.loc[:,group_col]):
            data_by_groups = data.groupby(group_col)
            if scatter_groups_order is None:
                names=data_by_groups.groups.keys()
            else:
                names=scatter_groups_order
            # colors=sns.color_palette(scatter_groups_cmap, n_colors=len(names))
            # colors=['m','g','c','y']
            colors=[(1.0,1.0,0.0),
                    (0.4,1.0,0.4),
                    (0.0,1.0,1.0),
                   (1.0,0.0,0.8)]
            for groupi in range(len(names)):
                name=names[groupi]
                color=colors[groupi]
                group=data_by_groups.get_group(name)
                ax.plot(group.loc[:,x_col],group.loc[:,y_col],label=name,
                        marker='o', linestyle='', ms=ms, c=color,
                        alpha=scatter_groups_alpha,zorder=zorder)
            if scatter_groups_legend:
                if legend_loc=="out":
                    l=ax.legend(#loc='upper right', 
                                bbox_to_anchor=(1.55, 1.0),
                             frameon=True,
                             )
                    l.get_frame().set_facecolor('whitesmoke')
                    l.get_frame().set_edgecolor(None)
                else:
                    ax.legend(loc=legend_loc)
        else:
            s= ax.scatter(data.loc[:,x_col],data.loc[:,y_col], \
                          s=ms, c=data.loc[:,group_col],cmap=cmap,edgecolor='none',
                          alpha=scatter_groups_alpha,zorder=zorder)
    #         colorbar_ax = fig.add_axes([1, 0.15, 0.05, 0.8]) #[left, bottom, width, height]
            colorbar_ax = fig.add_axes([0.55, 0.15, 0.035, 0.8]) #[left, bottom, width, height]
            colorbar_ax2=fig.colorbar(s, cax=colorbar_ax)
            colorbar_ax2.set_label(cbar_label)
    # print contourf     
    if contourf:
        print contourlevels
        plot_contourf(data.loc[:,x_col],data.loc[:,y_col],data.loc[:,contour_col],
                     contourlevels,interp=contourinterp,
                      vmin=cbar_min,vmax=cbar_max,
                      cbar=cbar,cbar_label=cbar_label,
              xlabel=x_col,ylabel=y_col,fig=fig,ax=ax,
                      xlog=xlog,
             )        
    return ax 

def scatter_colorby_control(ax,x_ctrl,y_ctrl,x_lim,y_lim,
                           ctrl_label,alpha=0.05,
                            zorder_span=0,zorder_label=3):
    
    ax.axvspan(x_lim[0], x_ctrl, color='blue', alpha=alpha,zorder=zorder_span)
    ax.axvspan(x_ctrl, x_lim[1], color='red', alpha=alpha,zorder=zorder_span)
    ax.axhspan(y_lim[0], y_ctrl, color='blue', alpha=alpha,zorder=zorder_span)
    ax.axhspan(y_ctrl, y_lim[1], color='red', alpha=alpha,zorder=zorder_span)

    ax.axvline(x_ctrl,color='gray',alpha=0.5,zorder=zorder_span)
    ax.axhline(y_ctrl,color='gray',alpha=0.5,zorder=zorder_span)
    ax.text(x_ctrl,y_ctrl,ctrl_label,color='k',zorder=zorder_label)
    return ax
#corr withfeats


def get_medians(combo,col_median,col_unique):
    medians=pd.DataFrame(columns=[col_unique,col_median])
    medians[col_unique]=combo.loc[:,col_unique].unique()
    for i in medians.index:
        medians.loc[i,col_median]=combo.loc[combo.loc[:,col_unique]==medians.loc[i,col_unique],col_median].median()
    return medians

def correlate(data_all,cols,cols_lables,linear_reg=False,median_reg=False,color='k',plot_fh=None):
    import seaborn as sns
    sns.axes_style("whitegrid")
    combo=data_all.loc[:,cols]
    combo.columns=cols_lables    

    if median_reg!=False:        
        if median_reg=="x":
            medians=get_medians(combo,cols_lables[0],cols_lables[1])
        elif median_reg=="y":
            medians=get_medians(combo,cols_lables[1],cols_lables[0])
        from data2ml.lib.io_ml import denanrows
        medians=denanrows(medians)
        medians=medians.astype(float)
        medians_reg_m,medians_reg_c = np.polyfit(medians[cols_lables[0]], medians[cols_lables[1]], deg=1)    
    
    xlim=[combo.loc[:,cols_lables[0]].min(),combo.loc[:,cols_lables[0]].max()]
    ymin=combo.loc[:,cols_lables[1]].min()
    ymax=combo.loc[:,cols_lables[1]].max()
    ylen=ymax-ymin
    space=0.2
    ylim=[ymin,ymax+space*ylen]
    if linear_reg:
        ax=sns.jointplot(data=combo,x=cols_lables[0],y=cols_lables[1],
                         color=color,size=3.5,alpha=0.40,ratio=10,
    #                      kind="reg",
    #                      joint_kws={'line_kws':{'color':"r"}},
                         xlim=xlim,ylim=ylim,
                        )
    elif linear_reg==False:
        ax=sns.jointplot(data=combo,x=cols_lables[0],y=cols_lables[1],
                         color=color,size=3.5,alpha=0.40,ratio=12,
                         xlim=xlim,ylim=ylim,
                        )
    if median_reg!=False:        
#         ax.ax_joint.plot(medians[cols_lables[0]], medians_reg_m * medians[cols_lables[0]] + medians_reg_c,
#                          color='red')        
        ax.ax_joint.plot(np.array(xlim), medians_reg_m * np.array(xlim) + medians_reg_c,
                         color='red')
    if plot_fh!=None:
        plt.savefig(plot_fh,format="pdf")
#     print plot_fh
    return ax

def scatter(data_all,x_col,y_col,
    x_err_col,y_err_col,
    groups_cols,contour_cols,
            ctrl=None,x_ctrl=None,y_ctrl=None,ctrl_label=None,
            errorbar=True,scatter=True,scatter_groups=False,
            contourf=False,
            contourlevels=10,
            contourinterp='nn',
            ms=10,
            scatter_color="gray",scatter_alpha=0.5,
            scatter_groups_order=None,
            scatter_groups_cmap='YlGn',
            scatter_groups_legend=True,
            xlabel=None,ylabel=None,
            cbar=True,cbar_label="",cbar_min=None,cbar_max=None,cmap="Blues",
            title=None,
            yreverse=False,diag=False,legend_loc=0,space=0.05,
            xlog=False,ylog=False,logbase=10,
            minortickson=False,
            mplstyle='seaborn-white',
            set_xlims=[],
            set_ylims=[],
            figsize=(8,4),
            plot_fh=None,
           plot_formats=['pdf','png']):
    plt.style.use(mplstyle)
    if errorbar:
        data=data_all.loc[:,[x_col,y_col,x_err_col,y_err_col]+groups_cols+contour_cols]
    else:
        data=data_all.loc[:,[x_col,y_col]+groups_cols+contour_cols]
    data=denanrows(data)
    print np.shape(data)
    if ctrl:
        data.loc[len(data)+1,x_col]=x_ctrl
        data.loc[len(data),y_col]=y_ctrl
    X=np.array(data.loc[:,x_col])
    Y=np.array(data.loc[:,y_col])
    if errorbar:
        X_err=np.array(data.loc[:,x_err_col])
        Y_err=np.array(data.loc[:,y_err_col])
    print np.shape(X)
    xlims,ylims=get_axlims(X,Y,space=space)
    print ylims
    if len(set_xlims)==2:
        xlims=set_xlims
    if len(set_ylims)==2:
        ylims=set_ylims
#     print(plt.style.available)
    # plt.style.use('seaborn-white')#'classic')#'seaborn-white')#'fivethirtyeight')#'ggplot')
#     fig=plt.figure(figsize=(4,4),dpi=400)
#     ax=plt.subplot(111)
    fig=plt.figure(figsize=figsize,dpi=400)
    ax=plt.subplot(121)
    if ctrl:
        scatter_colorby_control(ax,x_ctrl,y_ctrl,x_lim,y_lim,
                                ctrl_label,zorder_span=0,zorder_label=3)
    for i in range(len(contour_cols)):
        # group_col = groups_cols[i]        
        contour_col = contour_cols[i]        
        ax=scatter_colorby_groups(data,x_col,y_col,
                                  group_col=None,
                                  contour_col=contour_col,
                                  scatter_groups=scatter_groups,
                                  contourf=contourf,
                                  contourlevels=contourlevels,
                                  contourinterp=contourinterp,
                                  scatter_groups_order=scatter_groups_order,
                                  scatter_groups_cmap=scatter_groups_cmap,
                                  scatter_groups_legend=scatter_groups_legend,
                                  ms=ms,
                                  cbar=cbar,cbar_label=cbar_label,
                                  cbar_min=cbar_min,cbar_max=cbar_max,
                                  cmap=cmap,
                                  xlog=xlog,
                                  ax=ax,fig=fig)
    for i in range(len(groups_cols)):
        group_col = groups_cols[i]        
        ax=scatter_colorby_groups(data,x_col,y_col,
                                  group_col=group_col,
                                  contour_col=None,
                                  scatter_groups=scatter_groups,
                                  contourf=contourf,
                                  scatter_groups_order=scatter_groups_order,
                                  scatter_groups_cmap=scatter_groups_cmap,
                                  scatter_groups_legend=scatter_groups_legend,
                                  ms=ms,
                                  cbar=cbar,cbar_label=cbar_label,cbar_min=cbar_min,cbar_max=cbar_max,
                                  cmap=cmap,
                                  ax=ax,fig=fig)
    if errorbar:
        ax.errorbar(X,Y,xerr=X_err,yerr=Y_err,fmt="none",ecolor='gray',alpha=0.25, capthick=2,zorder=0)
    if scatter:
        ax.plot(X,Y,marker='o', linestyle='', ms=ms,c=scatter_color,alpha=scatter_alpha,zorder=1)
    
    if xlabel==None:
        xlabel=x_col
    if ylabel==None:
        ylabel=y_col
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if diag:
        lims=[np.min([xlims[0],ylims[0]]),np.max([xlims[1],ylims[1]])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims,lims,'k--',zorder=-1)
    else:
        ax.set_xlim(xlims)
        if yreverse:
            ax.set_ylim(ylims[::-1])
        else:
            ax.set_ylim(ylims)
    if minortickson:
        # Hide the right and top spines
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        # ax.yaxis.set_ticks_position('left')
        # ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(axis='x',which='minor',bottom='off')
        ax.tick_params(axis='y',which='minor',bottom='off')
        ax.minorticks_on()
    if xlog:
        ax.set_xscale('log',basex=logbase)
    if ylog:
        ax.set_yscale('log',basey=logbase)
    if title!=None:
        ax.set_title(title)
    saveplot(plot_fh,form='both')
    return ax

def get_cbarlims(data_plot,mn=0.25,mx=0.75,log=False):
    if log:
        data_plot=data_plot.apply(np.log10)
    return  float(pd.DataFrame(data_plot.unstack()).dropna().quantile(mn)),\
            float(pd.DataFrame(data_plot.unstack()).dropna().quantile(mx))
