import os
'''
Configuration File Used for Cityscapes Training & Evaluation
'''

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')

# Define the directory path you want to access
nyuv2_dir = os.path.join(tmpdir, 'nyuv2/new_data/nyu_v2/')


DATA_ROOT = nyuv2_dir
CROP_H = 321
CROP_W = 321
TASKS = ["seg", "sn"]
TASKS_NUM_CLASS = [40, 3]

LAMBDAS = [1, 20]
NUM_GPUS = 1
BATCH_SIZE = 32
MAX_ITERS = 20000
DECAY_LR_FREQ = 4000
DECAY_LR_RATE = 0.5

INIT_LR = 1e-3
WEIGHT_DECAY = 5e-4
IMAGE_SHAPE = (480, 640)

PRUNE_TIMES = 11
PRUNE_ITERS = [100] * PRUNE_TIMES

# --------------------------------------------------------------- #
END = 15000
INT = 50
PRUNE_RATE = 0.5
RETRAIN_EPOCH = 1000
RETRAIN_LR = 1e-5
