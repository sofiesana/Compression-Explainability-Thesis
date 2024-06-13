#!/bin/bash
#SBATCH --job-name=irof_dynamic
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=10GB

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

########### GET MODELS

# make a directory in the TMPDIR for the pruned models
mkdir -p $TMPDIR/pruned/dynamic/dynamic2

# extract baselinme model from scratch to TMPDIR/dynamic

tar xzf /scratch/$USER/pruned_models/dynamic/50/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2
tar xzf /scratch/$USER/pruned_models/dynamic/70/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2
tar xzf /scratch/$USER/pruned_models/dynamic/80/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2
tar xzf /scratch/$USER/pruned_models/dynamic/90/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2

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
python3 irof_seg_all.py --head all --method dynamic --task seg --irof mean --model_num 2

########### GET RESULTS

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/irof/all/seg/dynamic/dynamic2_${SLURM_JOBID}
tar czvf /scratch/$USER/irof/all/seg/dynamic/dynamic2_${SLURM_JOBID}/results.tar.gz $TMPDIR/results