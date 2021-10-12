import sys
import os
from subprocess import call

data_dir = '/scratch/camachoc/hbn'

sublist = sys.argv[1]
subs = open(sublist).read().splitlines()

for sub in subs:
    subdir = data_dir + '/' + sub + '/MNINonLinear/Results'
    sublog = data_dir + '/' + sub + '/logs'
    try:
        fmri = os.listdir(subdir)
        for seq in fmri:
            motderivs = subdir + '/' + seq + '/Movement_Regressors_dt.txt'
            ts = subdir + '/' + seq + '/' + seq + '_Atlas.dtseries.nii'
            queuefile_name = sublog + '/{0}_{1}_connproc.sh'.format(sub, seq) 
            qfile = open(queuefile_name, 'w')
            qfile.write('#! /bin/bash\n')
            qfile.write('#SBATCH --job-name={0}_{1}_connproc\n'.format(sub,seq))
            qfile.write('#SBATCH --output={0}/job%j_%x.out\n'.format(sublog))
            qfile.write('#SBATCH  --nodes=1\n')
            qfile.write('#SBATCH --ntasks-per-node=1\n')
            qfile.write('#SBATCH --mem=16gb\n')
            qfile.write('#SBATCH --time=24:00:00\n')
            qfile.write('#SBATCH --mail-type=END,FAIL\n')
            qfile.write('#SBATCH --mail-user=camachoc@wustl.edu\n')
            qfile.write('module load  fsl/6.0.4 workbench/1.5.0 freesurfer/7.1.1 python/3.8.3\n')
            qfile.write('source activate hcppython \n\n')
            qfile.write('python3 {0}/connproc.py {1} {2} \n'.format(data_dir, ts, motderivs))
            
            if 'movie' in seq:
                qfile.write('python3 {0}/actproc.py {1} {2} \n'.format(data_dir, ts, motderivs))

            qfile.close()
            call(['sbatch', queuefile_name])
    except:
        print('no fMRI data found for sub {0}'.format(sub))
