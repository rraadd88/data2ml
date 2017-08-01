#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_data_files``
================================
"""
# import sys
import pandas as pd
from os.path import exists,basename,abspath,dirname,expanduser
import logging
from glob import glob  
import numpy as np
# from data2ml.lib.io_seq_files import get_fsta_feats
logging.basicConfig(format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..): %(message)s',level=logging.DEBUG) # filename=cfg_xls_fh+'.log'

import pickle

## DEFS
def is_cfg_ok(cfg_dh,cfgs) :
    """
    Checks if the required files are present in given directory.
    
    :param cfg_dh: path to directory.
    :param cfgs: list of names of files.
    """
    cfg_dh_cfgs=glob(cfg_dh+"/*")
    cfg_dh_cfgs=[basename(cfg_dh_cfg) for cfg_dh_cfg in cfg_dh_cfgs]
    for cfg in cfgs :   # check if required sheets are present
        if not cfg in cfg_dh_cfgs :
            logging.error("%s does not exist" % cfg)    
            return False
            break
    return True

def auto_find_missing_paths(prj_dh):
    info=pd.read_csv(prj_dh+"/cfg/info")
    info_path_vars=[varn for varn in info['varname'] if ("_fh" in varn) or ("_dh" in varn)]
    info=info.set_index("varname")
    #find pdb_fh and fsta_fh in prj_dh
    if pd.isnull(info.loc["pdb_fh","input"]):
        try:
            info.loc["pdb_fh","input"]=glob("%s/*.pdb" % prj_dh)[0]
        except:
            logging.error("can not find .pdb file")
    if pd.isnull(info.loc["fsta_fh","input"]):
        try:
            fsta_fhs=glob("%s/*.fasta" % prj_dh)
            for fsta_fh in fsta_fhs:
                if 'prt' not in fsta_fh:
                    info.loc["fsta_fh","input"]=fsta_fh
                    break
        except:
            logging.error("could not find .fasta file")     
    info_paths=[info.loc[info_path_var,"input"] for info_path_var in info_path_vars]
    info.reset_index().to_csv(prj_dh+"/cfg/info",index=False)
    # if any(pd.isnull(info_paths)):
    if np.nan in info_paths:
        from data2ml import configure
        configure.main(prj_dh,"deps")  
    # return 

# from data2ml.lib.io_seq_files import cctmr_fasta2ref_fasta
def info2src(prj_dh):
    """
    This converts `.csv` configuration file to `.py` source file saved in `/tmp/`.
    
    :param prj_dh: path to project directory
    """        
    from data2ml.lib.io_seq_files import fasta_nts2prt
    auto_find_missing_paths(prj_dh)
    info=pd.read_csv(prj_dh+"/cfg/info")
    # info=auto_find_missing_paths(prj_dh)
    info_path_vars=[varn for varn in info['varname'] if ("_fh" in varn) or ("_dh" in varn)]
    info=info.set_index("varname")

    # find still missing paths ones
    info_paths=[info.loc[info_path_var,"input"] for info_path_var in info_path_vars]
    for info_path in info_paths:
        if not pd.isnull(info_path):
	     if not exists(info_path):
                if (info_path!='bowtie2') or (info_fh!='samtools'):
	            logging.error('Path to files do not exist. Include correct path in cfg/info. %s : %s' % (info_path_vars[info_paths.index(info_path)],info_path))
        	    return None
        else:
            logging.error('Path to file is missing. Check in cfg/info. %s : %s' % (info_path_vars[info_paths.index(info_path)],info_path))
            return None

    if not pd.isnull(info.loc['cctmr','input']):
        
        cctmr=info.loc['cctmr','input']
        cctmr=[int("%s" % i) for i in cctmr.split(" ")]
        # aas_len=cctmr[1]-1
        fsta_fh=cctmr_fasta2ref_fasta(info.loc['fsta_fh','input'],cctmr)
    else:

        fsta_fh=info.loc['fsta_fh','input']
    info.loc['prj_dh','input']=abspath(prj_dh)
    info.loc['fsta_id','input'],info.loc['fsta_seq','input'],info.loc['fsta_len','input']=get_fsta_feats(fsta_fh)
    host=info.loc['host','input']
    if pd.isnull(host):
        host=info.loc['host','default']    	
    info.loc['prt_seq','input']=fasta_nts2prt(fsta_fh,host=host).replace('*','X')
    info.reset_index().to_csv(prj_dh+"/cfg/info",index=False)
    csv2src(prj_dh+"/cfg/info","%s/../tmp/info.py" % (abspath(dirname(__file__))))
    logging.info("configuration compiled: %s/cfg/info" % prj_dh)

def csv2src(csv_fh,src_fh):
    """
    This writes `.csv` to `.py` source file.
    
    :param csv_fh: path to input `.csv` file.
    :param src_fh: path to output `.py` source file.
    """
    info=pd.read_csv(csv_fh)
    info=info.set_index('varname')    
    src_f=open(src_fh,'w')
    src_f.write("#!usr/bin/python\n")
    src_f.write("\n")
    src_f.write("# source file for data2ml's configuration \n")
    src_f.write("\n")

    for var in info.iterrows() :
        val=info['input'][var[0]]
        if pd.isnull(val):
            val=info['default'][var[0]]
        src_f.write("%s='%s' #%s\n" % (var[0],val,info["description"][var[0]]))
    src_f.close()

def raw_input2info(prj_dh,inputORdefault):     
    """
    This writes configuration `.csv` file from `raw_input` from prompt.
    
    :param prj_dh: path to project directory.
    :param inputORdefault: column name "input" or "default". 
    """
    info=pd.read_csv(prj_dh+"/cfg/info")
    info=info.set_index("varname",drop=True)
    for var in info.index.values:
        val=raw_input("%s (default: %s) =" % (info.loc[var, "description"],info.loc[var, "default"]))
        if not val=='':
            info.loc[var, inputORdefault]=val
    info.reset_index().to_csv("%s/cfg/info" % prj_dh, index=False)

def is_xls_ok(cfg_xls,cfg_xls_sheetnames_required) :
    """
    Checks if the required sheets are present in the configuration excel file.
    Input/s : path to configuration excel file 
    """
    cfg_xls_sheetnames=cfg_xls.sheet_names
    cfg_xls_sheetnames= [str(x) for x in cfg_xls_sheetnames]# unicode to str

    for qry_sheet_namei in cfg_xls_sheetnames_required :   # check if required sheets are present
        #qry_sheet_namei=str(qry_sheet_namei)
        if not qry_sheet_namei in cfg_xls_sheetnames :
            logging.error("pipeline : sheetname '%s' does not exist" % qry_sheet_namei)    
            return False
            break
    return True

def is_info_ok(xls_fh):
    """
    This checks the sanity of info sheet in the configuration excel file.
    For example if the files exists or not.
    """
    info=pd.read_excel(xls_fh,'info')
    info_path_vars=[varn for varn in info['varname'] if ("_fh" in varn) or ("_dh" in varn)]
    info=info.set_index("varname")
    info_paths=[info.loc[info_path_var,"input"] for info_path_var in info_path_vars]
    for info_path in info_paths:
        if not pd.isnull(info_path): 
            if not exists(info_path):                
                return False #(info_path_vars[info_paths.index(info_path)],info_path)
                break
    return True    

def xls2h5(cfg_xls,cfg_h5,cfg_xls_sheetnames_required) :
    """
    Converts configuration excel file to HDF5(h5) file.
    Here sheets in excel files are converted to groups in HDF5 file.
    Input/s : (path to configuration excel file, path to output HDF5 file)
    """
    for qry_sheet_namei in  cfg_xls_sheetnames_required:  
        qry_sheet_df=cfg_xls.parse(qry_sheet_namei)
        qry_sheet_df=qry_sheet_df.astype(str) # suppress unicode error
        qry_sheet_df.columns=[col.replace(" ","_") for col in qry_sheet_df.columns]
        cfg_h5.put("cfg/"+qry_sheet_namei,convert2h5form(qry_sheet_df), format='table', data_columns=True)
    return cfg_h5

def xls2csvs(cfg_xls,cfg_xls_sheetnames_required,output_dh):
    """
    Converts configuration excel file to HDF5(h5) file.
    Here sheets in excel files are converted to groups in HDF5 file.
    Input/s : (path to configuration excel file, path to output HDF5 file)
    """
    for qry_sheet_namei in  cfg_xls_sheetnames_required:  
        qry_sheet_df=cfg_xls.parse(qry_sheet_namei)
        qry_sheet_df=qry_sheet_df.astype(str) # suppress unicode error
        qry_sheet_df.to_csv("%s/%s" % (output_dh,qry_sheet_namei))
#         print "%s/%s" % (output_dh,qry_sheet_namei)

def convert2h5form(df):
    from data2ml.lib.io_strs import convertstr2format 
    df.columns=[convertstr2format(col,"^[a-zA-Z0-9_]*$") for col in df.columns.tolist()]
    return df

def csvs2h5(dh,sub_dh_list,fn_list,output_dh,cfg_h5):
    """
    This converts the csv files to tables in HDF5.
    """
    for fn in fn_list:
        for sub_dh in sub_dh_list : # get aas or cds  
            fh=output_dh+"/"+dh+"/"+sub_dh+"/"+fn+""
            df=pd.read_csv(fh) # get mat to df
            df=df.loc[:,[col.replace(" ","_") for col in list(df.columns) if not (('index' in col) or ('Unnamed' in col)) ]]
            exec("cfg_h5.put('%s/%s/%s',df, format='table', data_columns=True)" % (dh,sub_dh,str(fn)),locals(), globals()) # store the otpts in h5 eg. cds/N/lbl        
            # print("cfg_h5.put('%s/%s/%s',df.convert_objects(), format='table', data_columns=True)" % (dh,sub_dh,str(fn))) # store the otpts in h5 eg. cds/N/lbl        

def csvs2h5(dh,sub_dh_list,fn_list):
    """
    This converts csvs into HDF5 tables.
    """
    for fn in fn_list:
        for sub_dh in sub_dh_list : # get aas or cds  
            fh=output_dh+"/"+dh+"/"+sub_dh+"/"+fn+""
            key=dh+"/"+sub_dh+"/"+fn
            if (exists(fh)) and (key in cfg_h5):
                df=pd.read_csv(fh) # get mat to df
                key=key+"2"
                cfg_h5.put(key,df.convert_objects(), format='table', data_columns=True) # store the otpts in h5 eg. cds/N/lbl        


#mut_lbl_fit_comparison
def getusable_lbls_list(prj_dh):
    """
    This detects the samples that can be processed.
    
    :param prj_dh: path to project directory.
    :returns lbls_list: list of names of samples that can be processed.
    """
    lbls=pd.read_csv(prj_dh+'/cfg/lbls')
    lbls=lbls.set_index('varname')
    lbls_list=[]
    #data_lbl cols: NiA mutids NiS NiN NiNcut NiNcutlog NiScut NiScutlog NiAcut NiAcutlog    
    for lbli,lbl in lbls.iterrows() :
        # print "%s/data_lbl/%s/%s" % (prj_dh,'aas',str(lbli))        
        if (not exists("%s/data_lbl/%s/%s" % (prj_dh,'aas',str(lbli)))):
            fh_1=expanduser(str(lbl['fhs_1']))
            lbl_mat_mut_cds_fh=[fh for fh in glob(fh_1+"*") if '.mat_mut_cds' in fh]
            if len(lbl_mat_mut_cds_fh)!=0:
                lbl_mat_mut_cds_fh=lbl_mat_mut_cds_fh[0]
                lbls_list.append([lbli,lbl_mat_mut_cds_fh])
            else :
                fh_1="%s/data_mutmat/%s" % (prj_dh,basename(fh_1))
                # print fh_1
                lbl_mat_mut_cds_fh=[fh for fh in glob(fh_1+"*") if '.mat_mut_cds' in fh]
                if len(lbl_mat_mut_cds_fh)!=0:
                    lbl_mat_mut_cds_fh=lbl_mat_mut_cds_fh[0]
                    lbls_list.append([lbli,lbl_mat_mut_cds_fh])
                else:    
                    logging.warning("can not find: %s" % fh_1)
        # else:
            # logging.info("already processed: %s" % (str(lbli)))
    return lbls_list

def getusable_fits_list(prj_dh,data_fit_dh='data_fit'):
    """
    This gets the list of samples that can be processed for fitness estimations.
    
    :param prj_dh: path to project directory.
    :returns fits_pairs_list: list of tuples with names of input and selected samples.
    """
    if exists('%s/cfg/fit'% (prj_dh)):
        fits=pd.read_csv(prj_dh+'/cfg/fit')

        if "Unnamed: 0" in fits.columns:
            fits=fits.drop("Unnamed: 0", axis=1)
        fits_pairs_list=[]
        sel_cols=[col for col in fits.columns.tolist() if "sel_" in col]
        for pairi in fits.index.values :
            unsel_lbl=fits.loc[pairi,"unsel"]
            sels=list(fits.loc[pairi,sel_cols])
            # print sels
            for sel_lbl in sels :
                if not pd.isnull(sel_lbl):
                    fit_lbl=sel_lbl+"_WRT_"+unsel_lbl
                    if (not exists("%s/%s/%s/%s" % (prj_dh,data_fit_dh,'aas',fit_lbl))):
                        fits_pairs_list.append([unsel_lbl,sel_lbl])
                    else :
                        logging.info("already processed: %s" % (fit_lbl))
        return fits_pairs_list
    else:    
        logging.warning("ana3_mutmat2fit : getusable_fits_list : not fits in cfg/fit")
        return []

def getusable_comparison_list(prj_dh):
    """
    This converts the table of tests and controls in configuration file into tuples of test and control.
    
    :param prj_dh: path to project directory.
    """
    comparisons=pd.read_csv(prj_dh+'/cfg/comparison')
    comparisons=comparisons.set_index('ctrl')
    comparison_list=[]
    for ctrl,row in comparisons.iterrows() :
        row=row[~row.isnull()]
        for test in row[0:] :
            comparison_list.append([ctrl,test])
    return  comparison_list  

def to_pkl(data,fh):
    if not fh is None:
        with open(fh, 'wb') as f:
            pickle.dump(data, f, -1)    

def read_pkl(fh):
    with open(fh,'rb') as f:
        return pickle.load(f) 
