import pandas as pd
import numpy as np
import pdb

def read_file(filename, longitudinal=False, rare=True):
    """read in EHR data."""
    xd_name = filename + '_demographics.csv'
    if not longitudinal:
        xc_name = filename + '_common_median_imputed.csv'
    xr_name = filename + '_rare.csv'
    label_name = filename + '_class.csv'

    xd = pd.read_csv(xd_name,index_col='PT_ID')
    if not longitudinal: 
        xc = pd.read_csv(xc_name,index_col='PT_ID')
    if rare:
        xr = pd.read_csv(xr_name,index_col='PT_ID')
    
    label = pd.read_csv(label_name,index_col='PT_ID')
    
    print('longitudinal =',longitudinal,'rare =',rare)

    if not longitudinal and rare:   # demographics, common, and rare labs
        df_X = pd.concat([xd, xc, xr],axis=1)
        print('loading demographics, common, and rare labs')
    elif not longitudinal:  # keep common labs in there, remove rare
        df_X = pd.concat([xd, xc],axis=1)
        print('loading demographics and common labs (rare = ',rare,')')
    elif not rare:  # if longitudinal AND don't include rare, use only demographics
        df_X = xd
        print('loading demographics only (longitudinal = ',longitudinal,')')
    else:   # for longitudinal case with rare, remove common labs, include everything else
        df_X = pd.concat([xd, xr],axis=1)
        print('loading demographics and rare labs (longitudinal = ',longitudinal,')')
    
    assert(all(df_X.index==label.index))
    ###
    # Drop total cholesterol (sorry for the hack)
    if '2093-3' in df_X.columns:
        print('dropping total cholesterol')
        df_X = df_X.drop('2093-3',axis=1)

    feature_names = np.array([x for x in df_X.columns.values if x != 'class'])

    X = df_X.values #.astype(float)
    y = label.values.flatten()
    pt_ids = df_X.index.values

    assert(X.shape[1] == feature_names.shape[0])
    # pdb.set_trace()
    return X, y, pt_ids, feature_names, filename + '_long_imputed.csv'
