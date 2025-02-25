#!/bin/bash
#SBATCH --job-name=explanations_sn_pt
#SBATCH --time=01:30:00
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
mkdir -p $TMPDIR/pruned/pt/pt1
mkdir -p $TMPDIR/pruned/pt/pt2
mkdir -p $TMPDIR/pruned/pt/pt3

# extract baselinme model from scratch to TMPDIR/pt

tar xzf /scratch/$USER/pruned_models/pt/50/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt1
tar xzf /scratch/$USER/pruned_models/pt/70/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt1
tar xzf /scratch/$USER/pruned_models/pt/80/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt1
tar xzf /scratch/$USER/pruned_models/pt/90/baseline1/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt1

tar xzf /scratch/$USER/pruned_models/pt/50/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt2
tar xzf /scratch/$USER/pruned_models/pt/70/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt2
tar xzf /scratch/$USER/pruned_models/pt/80/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt2
tar xzf /scratch/$USER/pruned_models/pt/90/baseline2/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt2

tar xzf /scratch/$USER/pruned_models/pt/50/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt3
tar xzf /scratch/$USER/pruned_models/pt/70/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt3
tar xzf /scratch/$USER/pruned_models/pt/80/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt3
tar xzf /scratch/$USER/pruned_models/pt/90/baseline3/trained_results.tar.gz -C $TMPDIR/pruned/pt/pt3

########### GET CODE

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
python3 get_all_explanations_hbrk.py --method pt --task sn

########### GET RESULTS

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/explanations/all/sn/pt/job_${SLURM_JOBID}
tar czvf /scratch/$USER/explanations/all/sn/pt/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results