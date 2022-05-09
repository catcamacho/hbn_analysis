#! /bin/python

import sys
import os
from subprocess import call

data_dir = '/scratch/camachoc/hbn/activation'
out_dir = '/scratch/camachoc/hbn/firstlevel'

dmA_ratings = data_dir + '/movieDM_regressors_setA.csv'
dmB_ratings = data_dir + '/movieDM_regressors_setB.csv'
tpA_ratings = data_dir + '/movieTP_regressors_setA.csv'
tpB_ratings = data_dir + '/movieTP_regressors_setB.csv'

sublist = sys.argv[1]
subs = open(sublist).read().splitlines()

for sub in subs:
    dm_ts = data_dir + '/' + seq + '_task-movieDM_bold1_AP_Atlas_rescale_resid0.9_filt_gordon.32k_fs_LR.ptseries.nii'
    tp_ts = data_dir + '/' + seq + '_task-movieTP_bold1_AP_Atlas_rescale_resid0.9_filt_gordon.32k_fs_LR.ptseries.nii'
    dmA_prefix = out_dir + '/{0}_movieDM_setA_'.format(sub)
    dmB_prefix = out_dir + '/{0}_movieDM_setB_'.format(sub)
    tpA_prefix = out_dir + '/{0}_movieTP_setA_'.format(sub)
    tpB_prefix = out_dir + '/{0}_movieTP_setB_'.format(sub)
    
    queuefile_name = out_dir + '/{0}_firstlevel.sh'.format(sub) 
    qfile = open(queuefile_name, 'w')
    qfile.write('#! /bin/bash\n')
    qfile.write('#SBATCH --job-name={0}_{1}_firstlevel\n'.format(sub,seq))
    qfile.write('#SBATCH --output={0}/job%j_%x.out\n'.format(sublog))
    qfile.write('#SBATCH  --nodes=1\n')
    qfile.write('#SBATCH --ntasks-per-node=1\n')
    qfile.write('#SBATCH --mem=16gb\n')
    qfile.write('#SBATCH --time=6:00:00\n')
    qfile.write('#SBATCH --mail-type=END,FAIL\n')
    qfile.write('#SBATCH --mail-user=camachoc@wustl.edu\n')
    qfile.write('module load python/3.8.3\n')
    qfile.write('source activate hcppython \n\n')
    
    if os.path.isfile(dm_ts):
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2}\n'.format(dm_ts, dmA_ratings, dmA_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2}\n'.format(dm_ts, dmB_ratings, dmB_prefix))
    
    if os.path.isfile(tp_ts):
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2}\n'.format(tp_ts, tpA_ratings, tpA_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2}\n'.format(tp_ts, tpB_ratings, tpB_prefix))

    qfile.close()
    call(['sbatch', queuefile_name])
    print("submitted {0}".format(sub))
