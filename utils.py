import numpy as np

SBATCH_PATH = '/home/itaitz/EC-KitY'
CONFIG_FILE_FORMAT = 'json'

EVAL_TIMEOUT = 3600
FITNESS_ERROR_VALUE = -100_000

DATASET_PATH = 'datasets'

linear_gen_weight = lambda gen: gen + 1
square_gen_weight = lambda gen: (gen + 1) ** 2
exp_gen_weight = lambda gen: np.e ** (gen + 1)
log_gen_weight = lambda gen: np.log(gen + 1)
sqrt_gen_weight = lambda gen: (gen + 1) ** 0.5

MIN_PLAYER_SUM = 12
MIN_DEALER_CARD = 1
# 12 <= player sum <= 21 (10 states)
# 1 <= dealer card <= 10 (10 states)
# 2 states for usable ace
BLACKJACK_STATE_ACTION_SPACE_SHAPE = (10, 10, 2)

FROZEN_LAKE_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
FROZEN_LAKE_MAP_SIZE = len(FROZEN_LAKE_MAP)
HOLES = [i * FROZEN_LAKE_MAP_SIZE + j
         for i in range(FROZEN_LAKE_MAP_SIZE)
         for j in range(FROZEN_LAKE_MAP_SIZE)
         if FROZEN_LAKE_MAP[i][j] == 'H']
FROZEN_LAKE_STATES = FROZEN_LAKE_MAP_SIZE ** 2 - len(HOLES) - 1

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_sbatch_str(gen, n_individuals, device, job_id, sub_population_idx):
    general_config = f'''#!/bin/bash
################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --array=1-{n_individuals}
#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 6-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --mail-type=FAIL            ### send email when job ends or fails
#SBATCH --job-name {device[0]}job_{gen}_%a			### name of the job
#SBATCH --output=jobs/{device}/{job_id}/{gen}_{sub_population_idx}_%a.out			### output log for running job
#SBATCH --wait
'''

    gpu_config = f'''#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission
    #SBATCH --mem=200M				### ammount of RAM memory, allocating more than 60G requires IT team's permission
    '''

    cpu_config = f'''#SBATCH --cpus-per-task=6 # 6 cpus per task – use for multithreading, usually with --tasks=1
    #SBATCH --mem=100M				### ammount of RAM memory, allocating more than 60G requires IT team's permission
    '''

    problem = 'nn' if device == 'gpu' else 'blackjack'

    code_config = f'''echo "SLURM_JOBID"=$SLURM_JOBID
### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate ec_env 				### activate a conda environment, replace my_env with your conda environment
python "/sise/home/itaitz/EC-KitY/{problem}_evaluator.py" configs/{device}/{job_id}/{gen}_{sub_population_idx}.{CONFIG_FILE_FORMAT} $SLURM_ARRAY_TASK_ID
'''

    device_config = gpu_config if device == 'gpu' else cpu_config
    return general_config + device_config + code_config
