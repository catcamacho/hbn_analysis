#! /bin/python

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

ts = sys.argv[1]
ratings = pd.read_csv(sys.argv[2], index_col=None)
prefix = sys.argv[3]
out_type = sys.argv[4]

working_dir = os.path.dirname(ts)
os.chdir(working_dir)

def massuni_linGLM(ciftiseries_file, designmat, out_prefix, out_type):
    ss = StandardScaler()
    designmat = pd.DataFrame(ss.fit_transform(designmat.to_numpy()), designmat.index, designmat.columns)
    neuro_img = nib.load(ciftiseries_file)
    neuro_data = neuro_img.get_fdata()
    neuro_data = ss.fit_transform(neuro_data)
    beta_data = np.zeros((designmat.shape[1],neuro_data.shape[1]))
    for a in range(0,neuro_data.shape[1]):
        Y = neuro_data[:,a]
        results = sm.GLM(Y, designmat, family=sm.families.Gaussian()).fit()
        beta_data[:,a] = results.params

    bm = neuro_img.header.get_axis(1)
    sc = nib.cifti2.cifti2_axes.ScalarAxis(name=['beta'])
    for i, reg in enumerate(designmat.columns):
        temp_img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(beta_data[i,:], axis=0),(sc, bm))
        nib.save(temp_img, out_prefix+'{0}.{1}.nii'.format(reg, out_type))

# run the scripts
massuni_linGLM(ts, ratings, prefix, out_type)
