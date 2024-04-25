import numpy as np

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

# player position: 4x12, monster position: 3x12
CLIFF_WALKING_MAP_SHAPE = (4, 12)
CW_UP = 0
CW_RIGHT = 1
CW_DOWN = 2
CW_LEFT = 3
NUM_CLIFF_WALKING_STATES = 37
MONSTER_CLIFF_SPACE_SHAPE = (48, 36)
CLIFF_UNPLAYABLE_STATES = list(range(37, 48))


MONSTER_CLIFF_STATES = 1332     # 4*12*3*12 - 1*11*3*12

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
