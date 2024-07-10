#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --time=2:00:00
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

# make a directory in the TMPDIR for the pruned models
mkdir -p $TMPDIR/pruned/pt/pt1

# extract baselinme model from scratch to TMPDIR/pt

tar xzf /scratch/$USER/pruned_models/pt/80/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt1

########### GET CODE

# make a directory in the TMPDIR for quantus
mkdir $TMPDIR/quantus

# Copy code to $TMPDIR
cp -r /scratch/s4716671/github/Quantus-Thesis-Version $TMPDIR/quantus

# make a directory in the TMPDIR for the code
mkdir $TMPDIR/code

# Copy code to $TMPDIR
cp -r /scratch/$USER/github/Compression-Explainability-Thesis $TMPDIR/code

# Navigate to TMPDIR
cd $TMPDIR/code/Compression-Explainability-Thesis

# make results directory
mkdir $TMPDIR/results

########### RUN CODE

# Run training
python3 semseg_preds.py --head all --task seg --irof mean --method pt --ratio 80

########### GET RESULTS

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/seg_preds/pt1/80/job_${SLURM_JOBID}
tar czvf /scratch/$USER/seg_preds/pt1/80/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results