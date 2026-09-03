

# Set to True to enable parent evolution, False for random sampling
parent_evolution = True

#model and track configuration
MODEL_NAME = "simpleJoe02.pth"
TRACK_NAME = "crossRoad.png"

#car size and screen size
WIDTH = 1200
HEIGHT = 800
NUM_CARS = 100

#car settings
MAX_SPEED = 8.0
REVERSE_SPEED = 8.5
ACCELERATION = 0.15
FRICTION = 0.15
TURN_SPEED = 7.0

#sensors
show_sens = False
SENSOR_LENGTH = 400
SENSOR_ANGLES = [
    90,
    60,
    30,
    0,
    -30,
    -60,
    -90
]

#learning rate and max episode steps
LEARNING_RATE = 0.003
MAX_EPISODE_STEPS = 9000

#start position
START_X = 105
START_Y = 680
START_ANGLE = 20.0

#Rewards:
LAP_REWARD = 10.0
CHECKPOINT_REWARD = 5.0
CRASH_PENALTY = -2.0
SPEED_REWARD_MULTIPLIER = 0.1


#Evolution settings:
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.05
SURVIVOR_LIMIT = 3
SURVIVOR_EXTRA_STEPS = 500