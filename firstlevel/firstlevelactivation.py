#! /bin/python

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

ts = sys.argv[1]
ratings = pd.read_csv(sys.argv[2], index_col=0)
prefix = sys.argv[3]

working_dir = os.path.dirname(ts)
os.chdir(working_dir)

def massuni_linregress(ciftiseries_file, designmat, out_prefix):
    from sklearn.preprocessing import StandardScaler
    designmat_df = designmat
    ss = StandardScaler()
    designmat = ss.fit_transform(designmat.to_numpy())
    neuro_img = nib.load(ciftiseries_file)
    neuro_data = neuro_img.get_fdata()
    neuro_data = ss.fit_transform(neuro_data)
    beta_data = np.zeros((designmat.shape[1],neuro_data.shape[1]))
    for a in range(0,neuro_data.shape[1]):
        Y = neuro_data[:,a]
        X = np.linalg.pinv(designmat)
        beta_data[:,a] = np.dot(X, Y)

    bm = neuro_img.header.get_axis(1)
    sc = nib.cifti2.cifti2_axes.ScalarAxis(name=['beta'])
    for i, reg in enumerate(designmat_df.columns):
        temp_img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(beta_data[i,:], axis=0),(sc, bm))
        nib.save(temp_img, out_prefix+'{0}.pscalar.nii'.format(reg))        
        
# run the scripts
massuni_linregress(ts, ratings, prefix)