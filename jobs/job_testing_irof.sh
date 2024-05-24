#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=6GB

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

# make a directory in the TMPDIR for the pre-trained model
mkdir $TMPDIR/pt

# extract pre-trained model from scratch to TMPDIR/pt
# Change 'try' to match the folder containing the desired model
tar xzf /scratch/$USER/baseline/baseline1/results.tar.gz -C $TMPDIR/pt

# make a directory in the TMPDIR for the code
mkdir $TMPDIR/code

# Copy code to $TMPDIR
cp -r /scratch/$USER/github/Compression-Explainability-Thesis $TMPDIR/code

# make a directory in the TMPDIR for quantus
mkdir $TMPDIR/quantus

# Copy code to $TMPDIR
cp -r /scratch/s4716671/github/Quantus-Thesis-Version $TMPDIR/quantus

# Navigate to TMPDIR
cd $TMPDIR/code/Compression-Explainability-Thesis

# make results directory
mkdir $TMPDIR/results

# Run training
python3 irof.py --head all --pruned n --task seg

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/irof/job_${SLURM_JOBID}
tar czvf /scratch/$USER/irof/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results