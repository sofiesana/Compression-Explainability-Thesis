#!/bin/bash
#SBATCH --job-name=irof_static
#SBATCH --time=1-00:00:00
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

########### GET MODELS

# make a directory in the TMPDIR for the pruned models
mkdir -p $TMPDIR/pruned/static/static1
mkdir -p $TMPDIR/pruned/static/static2
mkdir -p $TMPDIR/pruned/static/static3

# extract baselinme model from scratch to TMPDIR/static

tar xzf /scratch/$USER/pruned_models/static/50/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1
tar xzf /scratch/$USER/pruned_models/static/70/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1
tar xzf /scratch/$USER/pruned_models/static/80/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1
tar xzf /scratch/$USER/pruned_models/static/90/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1

tar xzf /scratch/$USER/pruned_models/static/50/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2
tar xzf /scratch/$USER/pruned_models/static/70/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2
tar xzf /scratch/$USER/pruned_models/static/80/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2
tar xzf /scratch/$USER/pruned_models/static/90/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2

tar xzf /scratch/$USER/pruned_models/static/50/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3
tar xzf /scratch/$USER/pruned_models/static/70/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3
tar xzf /scratch/$USER/pruned_models/static/80/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3
tar xzf /scratch/$USER/pruned_models/static/90/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3

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
python3 irof_seg_all.py --head all --method static --task seg --irof mean

########### GET RESULTS

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/irof/all/seg/static/job_${SLURM_JOBID}
tar czvf /scratch/$USER/irof/all/seg/static/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results