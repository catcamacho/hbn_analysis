#! /bin/python3

'''
This script processes cifti dense timeseries data for using in parcel or vertex-wise activation
analyses.

INPUTS:
    ts : cifti dense timeseries data (e.g., "task-movieTP_bold1_AP_Atlas.dtseries.nii")
    motderivs : motion parameter first order derivatives (e.g., "Movement_Regressors_dt.txt")

OUTPUTS:
    procts : processed dense timeseries cifti
    parcelts : processed data by parcel of a specified atlas
'''

import os
import nibabel as nib
import pandas as pd
import numpy as np
from subprocess import check_call
from glob import glob
import sys

ts = sys.argv[1]
motderivs = sys.argv[2]

working_dir = os.path.dirname(ts)
os.chdir(working_dir)
prefix = os.path.basename(working_dir)

# study-specific variables
#atlas_dlabel = '/scratch3/CamachoCat/HBN/test/null_lL_WG33/Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii'
atlas_dlabel = '/scratch/camachoc/hbn/null_lL_WG33/Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii'
TR = nib.load(ts).header.get_axis(0).step # in seconds
threshold = 0.9 # FD threshold
lowpass = 0.1 # in Hz
highpass = 0.008 # in Hz

# resample data if needed before processing
if '32k' not in ts:
    check_call(['wb_command', '-cifti-resample',ts,'COLUMN', atlas_dlabel, 'COLUMN', 'ADAP_BARY_AREA', 'CUBIC', ts.replace('.dtseries','.32k_fs_LR.dtseries')])
    ts = ts.replace('.dtseries','.32k_fs_LR.dtseries')

# functions
def rescale_signal(ts):
    # pull GM timeseries
    img = nib.load(ts)
    data = (img.get_fdata()/1000)*1000
    img_ss = nib.cifti2.cifti2.Cifti2Image(data,(img.header.get_axis(0),img.header.get_axis(1)))
    nib.save(img_ss, ts.replace('.32k_fs_LR','_rescale.32k_fs_LR'))
    
    standard_ts = ts.replace('.32k_fs_LR','_rescale.32k_fs_LR')
    return(standard_ts)

def make_noise_ts(ts, motderivs, threshold=threshold):
    # pull mean GM timeseries
    img = nib.load(ts)
    globalsig = np.mean(img.get_fdata(), axis=1)
    globalsig = np.expand_dims(globalsig, axis=1)

    # load movement metrics
    motion = np.loadtxt(motderivs)[:,6:]
    fd = np.sum(np.absolute(motion[:,:3]),axis=1) + 50*(np.pi/180)*np.sum(np.absolute(motion[:,3:]),axis=1)

    # create timeseries of volumes to censor
    vols_to_censor = fd>threshold
    n_vols = np.sum(vols_to_censor)
    if n_vols > 0:
        spikes = np.zeros((len(fd),n_vols))
        b = 0
        for a in range(0,len(fd)):
            if vols_to_censor[a]==1:
                spikes[a,b] = 1
                b = b + 1
        np.savetxt('{0}_spikes.txt'.format(prefix),spikes.astype(int))
    else:
        file = open('{0}_spikes.txt'.format(prefix), 'w')
        file.write('')
    outlier_vols = os.path.abspath('{0}_spikes.txt'.format(prefix))

    # create volterra series of motion derivatives
    params = motion.shape[1]
    num_lags = 6
    leadlagderivs = np.zeros((len(fd),params*num_lags))
    for i in range(0,params):
        for j in range(0,num_lags):
            leadlagderivs[:,j+num_lags*i] =  np.roll(motion[:,i],shift=j, axis=0)
            leadlagderivs[:j,j+num_lags*i] = 0

    # combine nuissance into one array
    fd = np.expand_dims(fd,axis=1)
    nuissance = np.hstack((fd, leadlagderivs, globalsig))
    # add spikes if present
    if n_vols > 0:
        nuissance = np.hstack((nuissance, spikes))

    np.savetxt(motderivs.replace('Movement_Regressors_dt','nuissance_thresh{0}'.format(threshold)),nuissance)
    denoise_mat = os.path.abspath(motderivs.replace('Movement_Regressors_dt','nuissance_thresh{0}'.format(threshold)))
    return(denoise_mat, outlier_vols)

def denoise_ts(denoise_mat, ts, threshold=threshold):
    # load nuissance regressors and add a 1s column
    noise_mat = np.loadtxt(denoise_mat)
    onescol = np.ones((noise_mat.shape[0],1))
    noise_mat = np.hstack((noise_mat,onescol))

    # load data and preallocate output arrays
    func_data = nib.load(standard_ts).get_fdata()
    coefficients = np.zeros((noise_mat.shape[1],func_data.shape[1]))
    resid_data = np.zeros(func_data.shape)

    # perform voxel-wise matrix inversion
    for x in range(0,func_data.shape[1]):
        y = func_data[:,x]
        inv_mat = np.linalg.pinv(noise_mat)
        coefficients[:,x] = np.dot(inv_mat,y)
        yhat=np.sum(np.transpose(coefficients[:,x])*noise_mat,axis=1)
        resid_data[:,x] = y - np.transpose(yhat)

    # make cifti header to save residuals and coefficients
    ax1 = nib.load(standard_ts).header.get_axis(0)
    ax2 = nib.load(standard_ts).header.get_axis(1)
    header = (ax1,ax2)
    # save outputs
    resid_image = nib.cifti2.cifti2.Cifti2Image(resid_data, header)
    resid_image.to_filename(standard_ts.replace('.32k_fs_LR','_resid{0}.32k_fs_LR'.format(threshold)))

    ax1.size = noise_mat.shape[1]
    header = (ax1, ax2)
    coeff_image = nib.cifti2.cifti2.Cifti2Image(coefficients, header)
    coeff_image.to_filename(standard_ts.replace('.32k_fs_LR','_denoisecoeff{0}.32k_fs_LR'.format(threshold)))

    weights = standard_ts.replace('.32k_fs_LR','_denoisecoeff{0}.32k_fs_LR'.format(threshold))
    denoised_ts = standard_ts.replace('.32k_fs_LR','_resid{0}.32k_fs_LR'.format(threshold))

    return(weights, denoised_ts)

def bandpass_ts(denoised_ts, lowpass, highpass, TR):
    # load data and preallocate output
    func_data = nib.load(denoised_ts).get_fdata().T
    filt_data = np.zeros(func_data.shape)

    sampling_rate = 1/TR
    n_timepoints = func_data.shape[1]
    F = np.zeros((n_timepoints))

    lowidx = int(np.round(lowpass / sampling_rate * n_timepoints))
    highidx = int(np.round(highpass / sampling_rate * n_timepoints)) 
    F[highidx:lowidx] = 1
    F = ((F + F[::-1]) > 0).astype(int)
    filt_data = np.real(np.fft.ifftn(np.fft.fftn(func_data) * F))
    
    # make cifti header to save filtered data
    ax1 = nib.load(denoised_ts).header.get_axis(0)
    ax2 = nib.load(denoised_ts).header.get_axis(1)
    header = (ax1,ax2)
    # make and save image
    filt_image = nib.cifti2.cifti2.Cifti2Image(filt_data.T, header)
    filt_image.to_filename(denoised_ts.replace('.32k_fs_LR','_filt.32k_fs_LR'))
    filtered_ts = denoised_ts.replace('.32k_fs_LR','_filt.32k_fs_LR')
    
    return(filtered_ts)


#### run the data! ####
standard_ts = rescale_signal(ts)
denoise_mat, outlier_vols =  make_noise_ts(standard_ts, motderivs)
weights, denoised_ts = denoise_ts(denoise_mat, standard_ts)
filtered_ts = bandpass_ts(denoised_ts, lowpass, highpass, TR)
print('final dense data saved at: ' + filtered_ts)

# apply gordon parcels to the timeseries
check_call(['wb_command','-cifti-parcellate', filtered_ts, atlas_dlabel,
            'COLUMN', filtered_ts.replace('.32k_fs_LR.dtseries','_gordon.32k_fs_LR.ptseries'), '-only-numeric'])

print('final parcel timeseries data saved at: ' + filtered_ts.replace('.32k_fs_LR.dtseries','_gordon.32k_fs_LR.ptseries'))

