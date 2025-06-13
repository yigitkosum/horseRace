import math
import pygame

# ——— EKLENTİLER ——— #
DESIRED_TRAITS = {           # <— hedef fenotipler
    "prey": {
        "speed_target": 2.2,
        "vision_target": 70,
    },
    "predator": {
        "speed_target": 3.5,
        "vision_target": 110,
    },
}
# RL sabitleri
RL_ALPHA   = 2e-3         # öğrenme hızı
GAMMA      = 0.98         # ödül iskontosu
REWARD_REPRO_THRESHOLD = 250.0
# ———————————— DEVAMI (eski ayarlar) ——————————— #
WIDTH, HEIGHT = 1700, 1000
NUM_AGENTS = 100
NUM_FOOD   = 200
MUTATION_RATE    = 0.1
FOOD_RESPAWN_RATE = 0.10
FPS = 60
EXTRA_INPUTS = 6
RAY_COUNT  = 9
RAY_LENGTH = 150
RAY_SPREAD = math.pi/3
GRID_SIZE  = 50
WHITE  = (255,255,255);  BLACK=(0,0,0)
GREEN  = (136,198,35);   BLUE =(0,100,255)
RED    = (255,0,0);      ORANGE=(255,165,0)
BEIGE  = (245,245,220)
PREY_REPRO_THRESHOLD  = 200
PREY_REPRO_COST       = 100
PRED_REPRO_THRESHOLD  = 600
PRED_REPRO_COST       = 350
STRESS_ENERGY_THRESHOLD = 50
STRESS_MUTATION_MULTIPLIER = 3.0
SPEED_PENALTY_THRESHOLD = 2.5
SPEED_PENALTY_RATE      = 0.05
VISION_PENALTY_THRESHOLD = 80
VISION_PENALTY_RATE      = 0.02
