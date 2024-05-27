#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=03:30:00
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

############ GETTING NYUV2 DATA:

# make a directory in the TMPDIR for the dataset
mkdir $TMPDIR/nyuv2

# extract data from scratch to TMPDIR/nyuv2
tar xzf /scratch/$USER/nyuv2/nyu_v2_with_val.tar.gz -C $TMPDIR/nyuv2

############ GETTING CODE FROM REPOSITORY:

# make a directory in the TMPDIR for the code
mkdir $TMPDIR/code

# Copy code to $TMPDIR
cp -r /scratch/$USER/github/Compression-Explainability-Thesis $TMPDIR/code

# make results directory
mkdir $TMPDIR/results

############ RUNNING:

# Navigate to TMPDIR
cd $TMPDIR/code/Compression-Explainability-Thesis

# Run training
python3 launch_training.py --dataset nyuv2 --method disparse_dynamic --ratio 50 --dest $TMPDIR/results --source $TMPDIR/pt

############ SAVING:

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/pruned_models/dynamic/50/dynamic1
tar czvf /scratch/$USER/pruned_models/dynamic/50/dynamic1/trained_results.tar.gz $TMPDIR/results