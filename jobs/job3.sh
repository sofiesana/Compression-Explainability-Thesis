#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=12GB

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.8.16-GCCcore-11.2.0   
 
# activate virtual environment
source $HOME/venvs/first/bin/activate

# make a directory in the TMPDIR for the dataset
mkdir $TMPDIR/nyuv2

# extract data from scratch to TMPDIR/nyuv2
tar xzf /scratch/$USER/nyuv2/nyu_v2_with_val.tar.gz -C $TMPDIR/nyuv2

# Copy code to $TMPDIR
cp -r /scratch/$USER/disparse $TMPDIR

# Navigate to TMPDIR
cd $TMPDIR/disparse

# make results directory
mkdir $TMPDIR/results

# Run training
python3 launch_training.py --dataset nyuv2 --method baseline --dest $TMPDIR/results

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/baseline/job_${SLURM_JOBID}
tar czvf /scratch/$USER/baseline/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results