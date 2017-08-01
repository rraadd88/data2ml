#!usr/bin/python

# Copyright 2016, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_strs``
================================
"""


def isstrallowed(s,form):
    """
    Checks is input string conforms to input regex (`form`).

    :param s: input string.
    :param form: eg. for hdf5: `"^[a-zA-Z_][a-zA-Z0-9_]*$"`
    """
    import re
    match = re.match(form,s)
    return match is not None

def convertstr2format(col,form):
    """
    Convert input string to input regex (`form`).
    
    :param col: input string.
    :param form: eg. for hdf5: `"^[a-zA-Z_][a-zA-Z0-9_]*$"`
    """
    if not isstrallowed(col,form):
        col=col.replace(" ","_") 
        if not isstrallowed(col,form):
            chars_disallowed=[char for char in col if not isstrallowed(char,form)]
            for char in chars_disallowed:
                col=col.replace(char,"_")
    return col

def make_pathable_string(s):
    return s.replace(" ","_").replace("$","").replace("\\","")\
            .replace("(","").replace(")","")\
            .replace("{","").replace("}","").replace("%","_")\
            .replace(":","").replace("^","").replace("+","").replace("'","").replace("\"","")\
            .replace("\n","_").replace("\t","_")

def get_logger(argv=None):
    import logging
    import datetime
    log_format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'
    #'[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'

    logging.basicConfig(format=log_format,
                        level=logging.DEBUG,)
    if not argv is None:
        log_fh="%s_%s" % (make_pathable_string(str(datetime.datetime.now())),'_'.join(argv).replace('/','|'))
        print log_fh
        logging.basicConfig(filename=log_fh)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    # logging.info('#START')
    return logging
            
def linebreaker(l,break_pt=16):
    l_out=[]
    for i in l:
        if len(i)>break_pt:
            i_words=i.split(' ')
            i_out=''
            line_len=0
            for w in i_words:
                line_len+=len(w)+1
                if i_words.index(w)==0:
                    i_out=w
                elif line_len>break_pt:
                    line_len=0
                    i_out="%s\n%s" % (i_out,w)
                else:
                    i_out="%s %s" % (i_out,w)
            l_out.append(i_out)    
#             l_out.append("%s\n%s" % (i[:break_pt],i[break_pt:]))
        else:
            l_out.append(i)
    return l_out

def splitlabel(label,splitby=' ',ctrl='__'):
    splits=label.split(splitby)
    if len(splits)==2:
        return splits
    elif len(splits)==1:

        return splits+[ctrl]

def get_time():
    import datetime
    time=make_pathable_string('%s' % datetime.datetime.now())
    return time.replace('-','_').replace(':','_').replace('.','_')