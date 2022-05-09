import pandas as pd
import numpy as np
import nibabel as nib
import os
import scipy.stats as scp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.signal as scs
import json
import pickle
import plotly.graph_objects as go
from tqdm.auto import tqdm
from itertools import combinations
import statsmodels.formula.api as smf

sns.set(context='talk', style='white', font='Arial')

today = date.today().strftime('%Y%m%d')

project_dir = '/Users/catcamacho/Library/CloudStorage/Box-Box/CCP/HBN_study/'
data_dir = project_dir + 'proc/group/parcel_timeseries/sub_ts/'
out_dir = project_dir + 'proc/group/RSA/ISPS/'
sample_file = project_dir + 'proc/group/datasets_info/sample_gord.32k_fs_LR.pscalar.nii'
atlas_file = project_dir + 'proc/null_lL_WG33/Gordon333_SeitzmanSubcortical.32k_fs_LR.dlabel.nii'
os.makedirs(out_dir,exist_ok=True)

ax0 = nib.load(sample_file).header.get_axis(0)
ax1 = nib.load(sample_file).header.get_axis(1)

# load timeseries data info
subinfo = pd.read_csv(project_dir + 'proc/group/datasets/firstleveldatalabels_withpub_thresh0.8_20220412.csv', index_col=0)

# get network labels
parcel_labels = nib.load(sample_file).header.get_axis(1).name
network_labels = []
for s in parcel_labels:
    b = s.split('_')
    if len(b)<2:
        network_labels.append(b[0])
    else:
        network_labels.append(b[1])
network_labels = np.array(network_labels)
network_names, network_sizes = np.unique(network_labels, return_counts=True)

subinfo = subinfo.drop(['set','cond'], axis=1)
subinfo = subinfo.drop_duplicates()

# assign misc variables
TR = 0.8
niters = int(10000/len(parcel_labels))
alpha = np.sqrt(0.05/len(parcel_labels))

def compile_ts_data(subdf, movie, datadir, outfile):
    """
    combine data for each movie together into 1 file
    
    Parameters
    ----------
    subdf: DataFrame
        A dataframe with subject IDs as the index. Includes IDs for all usable data.
    movie: str
        Corresponds with the str for the movie content to concatenate (e.g., "DM" or "TP").
    datadir: folder path
        Path to folder with the subject timeseries ciftis.
    outfile: file path
        Path including filename to save the output data of shape Ntimepoints x Nparcels x Nsubjects.
    
    Returns
    -------
    data: numpy array
        The compiled data of shape Ntimepoints x Nparcels x Nsubjects
    """
    if not isinstance(subdf, pd.DataFrame):
        subdf = pd.read_csv(subdf, index_col=0)
    
    for sub in subdf.index:
        file = '{0}{1}_task-movie{2}_bold1_AP_Atlas_rescale_resid0.9_filt_gordonseitzman.32k_fs_LR.ptseries.nii'.format(datadir,sub, movie)
        if sub == subdf.index[0]:
            data = StandardScaler().fit_transform(nib.load(file).get_fdata())
            data = np.expand_dims(data, axis=2)
        else:
            t = StandardScaler().fit_transform(nib.load(file).get_fdata())
            t = np.expand_dims(t, axis=2)
            data = np.concatenate([data,t],axis=2)
    
    print('Compile data from {0} brain regions measured at {1} timepoints from {2} participants.'.format(data.shape[1],data.shape[0],data.shape[2]))
    np.save(outfile, data)
    return(data)


def intersubject_timeseries_correlation(data, outprefix, ax0=ax0, ax1=ax1):
    """
    Parameters
    ----------
    data: numpy array
        data in the shape of Ntimepoints x Nregions x Nsubjects
    outprefix: str
        name to save ISC data to
    
    Returns
    -------
    intersub_isc: numpy array
        intersubject spearman correlations in the shape of Nregions x Nsubjects x Nsubjects
    group_isc: numpy array
        group mean spearman correlations in the shape of Nregions
    """
    subs = range(0,data.shape[2])
    
    intersub_isc = np.zeros((data.shape[1],data.shape[2],data.shape[2]))
    group_isc = np.zeros((data.shape[1]))
    mask = np.tri(data.shape[2], data.shape[2], -1, dtype=int)
    
    for r in range(0, data.shape[1]):
        intersub_isc[r, :, :]= np.corrcoef(data[:, r, :], rowvar=False)
            
    for r in range(0, data.shape[1]):
        group_isc[r] = np.mean(intersub_isc[r,:,:][mask==1])
    
    np.save(outprefix + 'intersub_timeseries_ISC.npy', intersub_isc)
    img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(group_isc, axis=0), (ax0, ax1))
    nib.save(img, outprefix + 'mean_timseries_ISC.pscalar.nii')
    
    return(intersub_isc, group_isc)


def intersubject_distance(data, outfile_prefix):
    """
    Compute static pairwise intersubject similarity
    
    Parameters
    ----------
    data: numpy array
        1D array of subject data (i.e., each participant contributes exactly 1 measure)
    outfilename: str
        name to save distance data to
    
    Returns
    -------
    isdistances: numpy array
        intersubject distances in the shape of Nsubjects x Nsubjects x Nmetrics
    """
    subs = range(0,data.shape[0])


    # NN
    nn = np.zeros((data.shape[0],data.shape[0]))
    combs = itertools.combinations(subs, 2)
    for c in combs:
        nn[c[0],c[1]] = np.max(data) - abs(data[c[0]] - data[c[1]])
        nn[c[1],c[0]] = np.max(data) - abs(data[c[0]] - data[c[1]])
    np.save(outfile_prefix + '_NN.npy', nn)

    # AnnaK mean
    annakmean = np.zeros((data.shape[0],data.shape[0]))
    combs = itertools.combinations(subs, 2)
    for c in combs:
        annakmean[c[0],c[1]] = (data[c[0]] + data[c[1]]) / 2
        annakmean[c[1],c[0]] = (data[c[0]] + data[c[1]]) / 2
    np.save(outfile_prefix + '_annakmean.npy', annakmean)
    
    # AnnaK max min mean
    AnnaKmaxminmean = np.zeros((data.shape[0],data.shape[0]))
    combs = itertools.combinations(subs, 2)
    for c in combs:
        AnnaKmaxminmean[c[0],c[1]] = np.max(data) - ((data[c[0]] + data[c[1]]) / 2)
        AnnaKmaxminmean[c[1],c[0]] = np.max(data) - ((data[c[0]] + data[c[1]]) / 2)
    np.save(outfile_prefix + '_annakmaxminmean.npy', AnnaKmaxminmean)

    # AnnaK min
    annakmin = np.zeros((data.shape[0],data.shape[0]))
    combs = itertools.combinations(subs, 2)
    for c in combs:
        annakmin[c[0],c[1]] = min([data[c[0]],data[c[1]]])
        annakmin[c[1],c[0]] = min([data[c[0]],data[c[1]]])
    np.save(outfile_prefix + '_annakmin.npy', annakmin)

    # AnnaK max minus min
    annakmaxminmax = np.zeros((data.shape[0],data.shape[0]))
    combs = itertools.combinations(subs, 2)
    for c in combs:
        annakmaxminmax[c[0],c[1]] =np.max(data) -  max([data[c[0]],data[c[1]]])
        annakmaxminmax[c[1],c[0]] = np.max(data) - max([data[c[0]],data[c[1]]])
    np.save(outfile_prefix + '_annakmaxminmax.npy', annakmaxminmax)
        
    # AnnaK absmean
    annakabsmean = np.zeros((data.shape[0],data.shape[0]))
    combs = itertools.combinations(subs, 2)
    for c in combs:
        annakabsmean[c[0],c[1]] = abs(data[c[0]] - data[c[1]]) * ((data[c[0]] + data[c[1]]) / 2)
        annakabsmean[c[1],c[0]] = abs(data[c[0]] - data[c[1]]) * ((data[c[0]] + data[c[1]]) / 2)
    np.save(outfile_prefix + '_annakabsmean.npy', annakabsmean)
    
    isdistances = {'NN': nn, 
                   'AnnaKmean': annakmean, 
                   'AnnaKmin': annakmin, 
                   'AnnaKabsmean': annakabsmean, 
                   'AnnaKmaxminmean': AnnaKmaxminmean, 
                   'AnnaKmaxminmax': annakmaxminmax}
    return(isdistances)

def static_brain_bx_isrsa(brain_sim_data, bx_sim_data, outfilename=None):
    """
    
    Parameters
    ----------
    brain_sim_data: numpy ndarray
        Data in the shape of Nsubjects x Nsubjects
    bx_sim_data: numpy ndarray
        Data in the shape of Nsubjects x Nsubjects
        
    Returns
    -------
    rsa_report: pandas DataFrame
        Pandas DataFrame with inter-subject representational similarity statistics
    """
    rsa_report = pd.DataFrame(columns=['SpearR','SpearPvalue'])
    
    mask = np.tri(bx_sim_data.shape[0], bx_sim_data.shape[0], -1, dtype=int)
    bx_sim = bx_sim_data[mask==1]
    brain_sim = brain_sim_data[mask==1]
    
    r, p = scp.spearmanr(bx_sim, brain_sim)
    rsa_report.loc[0,'SpearR'] = r
    rsa_report.loc[0,'SpearPvalue'] = p
    if outfilename:
        sns.scatterplot(bx_sim, brain_sim)
        plt.title('Similarity Correlation')
        plt.tight_layout()
        plt.savefig(outfilename)
        plt.show()
        plt.close()
    
    return(rsa_report)


def regional_perm_bx_isrsa(regional_sim_data, bx_sim_data, outprefix, alpha=0.05, n_perms=1000, ax0=ax0, ax1=ax1):
    """
    
    Parameters
    ----------
    regional_sim_data: numpy ndarray
        Data in the shape of Nregions x Nsubjects x Nsubjects
    bx_sim_data: numpy ndarray
        Data in the shape of Nsubjects x Nsubjects
        
    Returns
    -------
    region_isrsa: numpy ndarray
        Data in the shape of Nregions
    """
    
    mask = np.tri(bx_sim_data.shape[1], bx_sim_data.shape[1], -1, dtype=int)

    # flatten behavior lower triangle
    bx_sim = bx_sim_data[mask==1]

    region_isrsa = np.zeros((regional_sim_data.shape[0]))

    for region in range(0, regional_sim_data.shape[0]):
            brain_sim = regional_sim_data[region,:,:][mask==1]
            r, p = scp.spearmanr(bx_sim, brain_sim)
            region_isrsa[region] = r

    shuff_bx = bx_sim
    perm_isrsa_null = np.zeros((n_perms, regional_sim_data.shape[0]))

    # make null distributions for each TR and region
    for a in range(0,n_perms):
        np.random.shuffle(shuff_bx)
        for region in range(0,regional_sim_data.shape[0]):
            brain_sim = regional_sim_data[region,:,:][mask==1]
            r, p = scp.spearmanr(shuff_bx, brain_sim)
            perm_isrsa_null[a, region] = r

    # compute permuted P threshold per region/TR
    raw_pvals = np.zeros(region_isrsa.shape)
    flat_null = perm_isrsa_null.flatten()
    for i, a in enumerate(region_isrsa):
        raw_pvals[i] = (np.sum((flat_null>=a).astype(int)) + 1) / (flat_null.shape[0] + 1)
        
    # save ciftis with raw values
    img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(raw_pvals, axis=0), (ax0, ax1))
    nib.save(img, outprefix + '_permsim_raw_pval.pscalar.nii')
    
    img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(region_isrsa, axis=0), (ax0, ax1))
    nib.save(img, outprefix + '_permsim_raw_rho.pscalar.nii')
    
    
    # save cifti with significant rhos only
    thresh_mask = raw_pvals<alpha

    # pvals
    thresh_pval = raw_pvals
    thresh_pval[thresh_mask==0] = np.nan
    img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(thresh_pval, axis=0), (ax0, ax1))
    nib.save(img, outprefix + '_permsim_masked_pval{0}.pscalar.nii'.format(alpha))

    # rhos
    thresh_isrsa = region_isrsa
    thresh_isrsa[thresh_mask==0] = np.nan
    thresh_isrsa[thresh_isrsa<0] = np.nan
    img = nib.cifti2.cifti2.Cifti2Image(np.expand_dims(thresh_isrsa, axis=0), (ax0, ax1))
    nib.save(img, outprefix + '_permsim_masked_rho{0}.pscalar.nii'.format(alpha))
    return(thresh_isrsa)


def region_isrsa_fdr(disc_rho, disc_pval, rep_rho, rep_pval, outprefix, alpha=0.05, bon_alpha=True,replace_zeros=True, ax0=ax0, ax1=ax1):
    """
    
    
    """
    
    disc_rho = nib.load(disc_rho).get_fdata()
    disc_pval = nib.load(disc_pval).get_fdata()
    rep_rho = nib.load(rep_rho).get_fdata()
    rep_pval = nib.load(rep_pval).get_fdata()
    
    if replace_zeros:
        disc_pval[disc_pval==0] = np.nan
        rep_pval[rep_pval==0] = np.nan
    
    if bon_alpha==True:
        bon_alpha = np.sqrt(alpha/disc_pval.shape[1])
    else:
        bon_alpha = alpha

    dmask = (disc_pval<bon_alpha).astype(int)
    rmask = (rep_pval<bon_alpha).astype(int)

    mask = np.zeros(dmask.shape)
    mask[(dmask==1) & (rmask==1)] = 1

    bonrho = np.empty(mask.shape)
    bonrho[mask==1] = np.add(disc_rho[mask==1],rep_rho[mask==1])/2
    bonrho[mask==0] = np.nan

    img = nib.cifti2.cifti2.Cifti2Image(bonrho, (ax0, ax1))
    nib.save(img, outprefix + '_maskedrho_fdr{0}.pscalar.nii'.format(round(bon_alpha,5)))
    
    
# process isc data 
sample = 'rubic'
movie = 'DM'

print(sample, movie)

sampleinfo = subinfo.loc[(subinfo['site']==sample) & (subinfo['movie']==movie),:]
outdir = os.path.join(out_dir, 'fullsample', 'ts_isc_{0}_movie{1}'.format(sample, movie))
os.makedirs(outdir, exist_ok=True)

group_data_file = os.path.join(outdir, 'compiled_timeseries_data_{0}_movie{1}.npy'.format(sample, movie))
if os.path.isfile(group_data_file):
    group_data = np.load(group_data_file)
else:
    group_data = compile_ts_data(sampleinfo, movie, data_dir, group_data_file)

outprefix = os.path.join(outdir, '{0}_movie{1}_'.format(sample, movie))
if os.path.isfile(outprefix + 'intersub_timeseries_ISC.npy'):
    regional_sim_data = np.load(outprefix + 'intersub_timeseries_ISC.npy')
else:
    regional_sim_data, mean_isc = intersubject_timeseries_correlation(group_data, outprefix)

# make null distribution
null_isc = np.zeros((niters, len(parcel_labels), len(sampleinfo), len(sampleinfo)))
for i in tqdm(range(0,niters)):
    group_data = np.load(group_data_file)
    orig_shape = group_data.shape
    group_data = group_data.flatten()
    np.random.shuffle(group_data)
    group_data = group_data.reshape(orig_shape)
    outprefix = os.path.join(outdir, 'null_{0}_movie{1}_'.format(sample, movie))
    null_isc[i,:,:,:], _ = intersubject_timeseries_correlation(group_data, outprefix)
np.save(outprefix + 'full_isc.npy', null_isc)
        