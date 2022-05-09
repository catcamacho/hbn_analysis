#! /bin/python

import sys
import os
from subprocess import call

data_dir = '/scratch/camachoc/hbn/activation'
reg_dir = '/scratch/camachoc/hbn/regressors'

dmG = reg_dir + '/movieDM_general.csv'
dmS = reg_dir + '/movieDM_specific.csv'
tpG = reg_dir + '/movieTP_general.csv'
tpS = reg_dir + '/movieTP_specific.csv'

sublist = sys.argv[1]
subs = open(sublist).read().splitlines()

for sub in subs:
    dm_ts = data_dir + '/{0}/{0}_task-movieDM_bold1_AP_Atlas_rescale_resid0.9_filt_gordon.32k_fs_LR.ptseries.nii'.format(sub)
    tp_ts = data_dir + '/{0}/{0}_task-movieTP_bold1_AP_Atlas_rescale_resid0.9_filt_gordon.32k_fs_LR.ptseries.nii'.format(sub)
    dm_dts = data_dir + '/{0}/{0}_task-movieDM_bold1_AP_Atlas_rescale_resid0.9_filt.32k_fs_LR.dtseries.nii'.format(sub)
    tp_dts = data_dir + '/{0}/{0}_task-movieTP_bold1_AP_Atlas_rescale_resid0.9_filt.32k_fs_LR.dtseries.nii'.format(sub)
    dmG_prefix = data_dir + '/{0}/{0}_movieDM_general_'.format(sub)
    dmS_prefix = data_dir + '/{0}/{0}_movieDM_specific_'.format(sub)
    tpG_prefix = data_dir + '/{0}/{0}_movieTP_general_'.format(sub)
    tpS_prefix = data_dir + '/{0}/{0}_movieTP_specific_'.format(sub)

    queuefile_name = data_dir + '/{0}_firstlevel.sh'.format(sub)
    qfile = open(queuefile_name, 'w')
    qfile.write('#! /bin/bash\n')
    qfile.write('#SBATCH --job-name={0}_firstlevel\n'.format(sub))
    qfile.write('#SBATCH --output={0}/job%j_%x.out\n'.format(data_dir))
    qfile.write('#SBATCH  --nodes=1\n')
    qfile.write('#SBATCH --ntasks-per-node=1\n')
    qfile.write('#SBATCH --mem=16gb\n')
    qfile.write('#SBATCH --time=6:00:00\n')
    qfile.write('#SBATCH --mail-type=END,FAIL\n')
    qfile.write('#SBATCH --mail-user=camachoc@wustl.edu\n')
    qfile.write('module load python/3.8.3\n')
    qfile.write('source activate hcppython \n\n')

    if os.path.isfile(dm_ts):
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} pscalar\n'.format(dm_ts, dmG, dmG_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} pscalar\n'.format(dm_ts, dmS, dmS_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} dscalar\n'.format(dm_dts, dmG, dmG_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} dscalar\n'.format(dm_dts, dmS, dmS_prefix))

    if os.path.isfile(tp_ts):
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} pscalar\n'.format(tp_ts, tpG, tpG_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} pscalar\n'.format(tp_ts, tpS, tpS_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} dscalar\n'.format(tp_dts, tpG, tpG_prefix))
        qfile.write('python3 /scratch/camachoc/hbn/firstlevelactivation.py {0} {1} {2} dscalar\n'.format(tp_dts, tpS, tpS_prefix))

    qfile.close()
    call(['sbatch', queuefile_name])
    print("submitted {0}".format(sub))
