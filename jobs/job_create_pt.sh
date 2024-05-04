#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.8.16-GCCcore-11.2.0   
 
# activate virtual environment
source $HOME/venvs/first/bin/activate

# make a directory in the TMPDIR for the pre-trained model
mkdir $TMPDIR/pt

# extract pre-trained model from scratch to TMPDIR/pt
# Change 'try' to match the folder containing the desired model
tar xzf /scratch/$USER/baseline/try/results.tar.gz -C $TMPDIR/pt

# Copy code to $TMPDIR
cp -r /scratch/$USER/disparse $TMPDIR

# Navigate to TMPDIR
cd $TMPDIR/disparse

# make results directory
mkdir $TMPDIR/results

# Run training
python3 create_pruned_net.py --dataset nyuv2 --method disparse_pt --ratio 90 --dest $TMPDIR/results


# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/pruned_models/pt/job_${SLURM_JOBID}
tar czvf /scratch/$USER/pruned_models/pt/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results