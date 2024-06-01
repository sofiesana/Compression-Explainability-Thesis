#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --time=00:15:00
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

# make a directory in the TMPDIR for the baseline model
mkdir -p $TMPDIR/baseline/baseline1
mkdir -p $TMPDIR/baseline/baseline2
mkdir -p $TMPDIR/baseline/baseline3

# extract baselinme model from scratch to TMPDIR/pt
tar xzf /scratch/$USER/baseline/baseline1/results.tar.gz -C $TMPDIR/baseline/baseline1
tar xzf /scratch/$USER/baseline/baseline2/results.tar.gz -C $TMPDIR/baseline/baseline2
tar xzf /scratch/$USER/baseline/baseline3/results.tar.gz -C $TMPDIR/baseline/baseline3

# make a directory in the TMPDIR for the pruned models
mkdir -p $TMPDIR/pruned/static/static1
mkdir -p $TMPDIR/pruned/static/static2
mkdir -p $TMPDIR/pruned/static/static3

# extract baselinme model from scratch to TMPDIR/pt
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

###### DYNAMIC

# make a directory in the TMPDIR for the pruned models
mkdir -p $TMPDIR/pruned/dynamic/dynamic1
mkdir -p $TMPDIR/pruned/dynamic/dynamic2
mkdir -p $TMPDIR/pruned/dynamic/dynamic3

# extract baselinme model from scratch to TMPDIR/dynamic

tar xzf /scratch/$USER/pruned_models/dynamic/50/dynamic1/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic1
tar xzf /scratch/$USER/pruned_models/dynamic/70/dynamic1/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic1
tar xzf /scratch/$USER/pruned_models/dynamic/80/dynamic1/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic1
tar xzf /scratch/$USER/pruned_models/dynamic/90/dynamic1/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic1

tar xzf /scratch/$USER/pruned_models/dynamic/50/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2
tar xzf /scratch/$USER/pruned_models/dynamic/70/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2
tar xzf /scratch/$USER/pruned_models/dynamic/80/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2
tar xzf /scratch/$USER/pruned_models/dynamic/90/dynamic2/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic2

tar xzf /scratch/$USER/pruned_models/dynamic/50/dynamic3/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic3
tar xzf /scratch/$USER/pruned_models/dynamic/70/dynamic3/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic3
tar xzf /scratch/$USER/pruned_models/dynamic/80/dynamic3/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic3
tar xzf /scratch/$USER/pruned_models/dynamic/90/dynamic3/trained_results.tar.gz -C $TMPDIR/pruned/dynamic/dynamic3

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
python3 irof_seg_all.py --head all --pruned n --task seg --irof mean

########### GET RESULTS

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/irof/all/seg/job_${SLURM_JOBID}
tar czvf /scratch/$USER/irof/all/seg/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results