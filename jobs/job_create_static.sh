#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=4GB

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

# make a directory in the TMPDIR for the code
mkdir $TMPDIR/code

# Copy code to $TMPDIR
cp -r /scratch/$USER/github/Compression-Explainability-Thesis $TMPDIR/code

# Navigate to TMPDIR
cd $TMPDIR/code/Compression-Explainability-Thesis

# make results directory
mkdir $TMPDIR/results

# Run training
python3 create_pruned_net.py --dataset nyuv2 --method disparse_static --ratio 50 --dest $TMPDIR/results

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/pruned_models/static/50/job_${SLURM_JOBID}
tar czvf /scratch/$USER/pruned_models/static/50/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results