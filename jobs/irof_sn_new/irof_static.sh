#!/bin/bash
#SBATCH --job-name=explanations_sn_static
#SBATCH --time=10:00:00
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
mkdir $TMPDIR/explanations

# extract data from scratch to TMPDIR/nyuv2
tar xzf /scratch/$USER/nyuv2/nyu_v2_with_val.tar.gz -C $TMPDIR/nyuv2
tar xzf /scratch/$USER/explanations/all/sn/static/job_15362006/results.tar.gz -C $TMPDIR/explanations

########### GET MODELS

# make a directory in the TMPDIR for the pruned models
mkdir -p $TMPDIR/pruned/static/static1
mkdir -p $TMPDIR/pruned/static/static2
mkdir -p $TMPDIR/pruned/static/static3

# extract baselinme model from scratch to TMPDIR/static

tar xzf /scratch/$USER/pruned_models/static/50/static1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1
tar xzf /scratch/$USER/pruned_models/static/70/static1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1
tar xzf /scratch/$USER/pruned_models/static/80/static1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1
tar xzf /scratch/$USER/pruned_models/static/90/static1/trained_results.tar.gz -C $TMPDIR/pruned/static/static1

tar xzf /scratch/$USER/pruned_models/static/50/static2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2
tar xzf /scratch/$USER/pruned_models/static/70/static2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2
tar xzf /scratch/$USER/pruned_models/static/80/static2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2
tar xzf /scratch/$USER/pruned_models/static/90/static2/trained_results.tar.gz -C $TMPDIR/pruned/static/static2

tar xzf /scratch/$USER/pruned_models/static/50/static3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3
tar xzf /scratch/$USER/pruned_models/static/70/static3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3
tar xzf /scratch/$USER/pruned_models/static/80/static3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3
tar xzf /scratch/$USER/pruned_models/static/90/static3/trained_results.tar.gz -C $TMPDIR/pruned/static/static3

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
python3 irof_sn_final_hbrk.py --method static --task sn

########### GET RESULTS

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/irof_new/sn/static/job_${SLURM_JOBID}
tar czvf /scratch/$USER/irof_new/sn/static/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results