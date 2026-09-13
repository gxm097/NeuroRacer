import pygame
import torch
import os
import numpy as np
import time

from car import Car
from Brain import CarBrain

from evolutionHandle import crossover

import config


from checkpoints import (
    find_checkpoints,
    find_lap_lines,
    marker_mask,
    CheckpointTracker,
    TREAT_RED_AS_ROAD,
    RED_ROAD_MARGIN
)


from trainer import (
    choose_action,
    action_to_controls,
    remember,
    train_model,
    clear_memory
)


pygame.init()


# ============================================================
# GAME SETTINGS
# ============================================================

screen = pygame.display.set_mode(
    (
        config.WIDTH,
        config.HEIGHT
    )
)


pygame.display.set_caption("ML Car Track Simulation")


clock = pygame.time.Clock()


font = pygame.font.SysFont("consolas", 26, bold=True)


small_font = pygame.font.SysFont(
    "consolas",
    16
)


# ============================================================
# LOAD TRACK
# ============================================================

TRACK_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "tracks"
)


TRACK_PATH = os.path.join(
    TRACK_DIR,
    config.TRACK_NAME
)


track_native = pygame.image.load(TRACK_PATH).convert()


track_image = pygame.transform.smoothscale(
    track_native,
    (
        config.WIDTH,
        config.HEIGHT
    )
)


collision_source = pygame.transform.scale(
    track_native,
    (
        config.WIDTH,
        config.HEIGHT
    )
)



# MODEL SAVE FILE
MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    config.MODEL_NAME
)


SAVE_EVERY_EPISODES = 25



# SAVE MODEL
def save_model(
    model,
    episode,
    optimizer=None,
    epsilon=0.0,
    filename=None
):

    if filename is None:

        filename = MODEL_PATH


    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )


    checkpoint = {
        "model_state":
            model.state_dict(),

        "epsilon":
            epsilon,

        "episode":
            episode
    }

    # DQN ONLY
    if optimizer is not None:

        checkpoint[
            "optimizer_state"
        ] = optimizer.state_dict()


    torch.save(
        checkpoint,
        filename
    )


    print(
        f"Model saved to {filename}"
    )

# LOAD MODEL
def load_model(
    model,
    filename=None,
    optimizer=None
):

    if filename is None:

        filename = MODEL_PATH


    checkpoint = torch.load(
        filename,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )


    # LOAD OPTIMIZER ONLY FOR DQN
    if (optimizer is not None and "optimizer_state" in checkpoint):
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )


    epsilon = checkpoint.get(
        "epsilon",
        0.0
    )


    episode = checkpoint.get(
        "episode",
        0
    )


    print(
        f"Loaded model from episode {episode}"
    )


    return epsilon, episode


# ============================================================
# ROAD DETECTION
# ============================================================

def build_drivable_array(
    surface
):

    rgb = pygame.surfarray.array3d(
        surface
    ).astype(
        np.int16
    )


    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]


    brightness = rgb.mean(
        axis=2
    )


    spread = (
        rgb.max(axis=2)
        -
        rgb.min(axis=2)
    )


    # --------------------------------------------------------
    # GRAY ASPHALT
    # --------------------------------------------------------

    gray = (
        (spread <= 25)
        &
        (brightness >= 60)
        &
        (brightness <= 215)
    )


    # --------------------------------------------------------
    # TAN
    # --------------------------------------------------------

    tan = (
        (r > g)
        &
        (g > b)
        &
        ((r - b) >= 25)
        &
        ((r - b) <= 110)
        &
        (brightness >= 140)
    )


    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    red = (
        (r >= 150)
        &
        (r >= g + 80)
        &
        (r >= b + 80)
    )


    # --------------------------------------------------------
    # BLUE
    # --------------------------------------------------------

    blue = (
        (b >= 150)
        &
        (b >= g + 60)
        &
        (b >= r + 60)
    )


    return (
        gray
        |
        tan
        |
        red
        |
        blue
    )


# ============================================================
# MASK DILATION
# ============================================================

def dilate(
    mask,
    radius
):

    out = mask.copy()


    for shift in range(
        -radius,
        radius + 1
    ):

        out |= np.roll(
            mask,
            shift,
            axis=0
        )


    wide = out.copy()


    for shift in range(
        -radius,
        radius + 1
    ):

        out |= np.roll(
            wide,
            shift,
            axis=1
        )


    return out


# ============================================================
# CLOSE SMALL GAPS
# ============================================================

def close_gaps(
    mask,
    radius
):

    return ~dilate(
        ~dilate(
            mask,
            radius
        ),
        radius
    )


LANE_MARKING_RADIUS = 0


drivable = close_gaps(
    build_drivable_array(
        collision_source
    ),
    LANE_MARKING_RADIUS
)


# ============================================================
# CHECKPOINTS
# ============================================================

_collision_rgb = pygame.surfarray.array3d(
    collision_source
).astype(
    np.int16
)


_markers = marker_mask(
    _collision_rgb
)


if TREAT_RED_AS_ROAD:

    drivable = (
        drivable
        |
        (
            _markers
            &
            dilate(
                drivable,
                RED_ROAD_MARGIN
            )
        )
    )


CHECKPOINTS = find_checkpoints(
    _collision_rgb
)


if CHECKPOINTS:

    print(
        f"Found {len(CHECKPOINTS)} "
        "checkpoint(s)."
    )

else:

    print(
        "No checkpoints found."
    )


LAP_LINES = find_lap_lines(
    _collision_rgb
)


print(
    f"Found {len(LAP_LINES)} "
    "lap line(s)."
)


# ============================================================
# CREATE ROAD MASK
# ============================================================

_mask_rgb = np.repeat(
    drivable[:, :, None],
    3,
    axis=2
).astype(
    np.uint8
) * 255


_mask_surface = (
    pygame.surfarray.make_surface(
        _mask_rgb
    )
)


road_mask = (
    pygame.mask.from_threshold(
        _mask_surface,
        (255, 255, 255),
        (50, 50, 50)
    )
)


# ============================================================
# KEEP ROAD CONNECTED TO START
# ============================================================

if road_mask.get_at(
    (
        config.START_X,
        config.START_Y
    )
):

    road_mask = (
        road_mask.connected_component(
            (
                config.START_X,
                config.START_Y
            )
        )
    )

else:

    print(
        "WARNING: start position "
        "is not on the road."
    )


# ============================================================
# OFF TRACK MASK
# ============================================================

off_track_mask = (
    road_mask.copy()
)


off_track_mask.invert()


# ============================================================
# CREATE CARS
# ============================================================

cars = []

checkpoint_trackers = []

episode_steps = []

active = []


for _ in range(
    config.NUM_CARS
):

    car = Car(
        config.START_X,
        config.START_Y,
        config.START_ANGLE,

        config.WIDTH,
        config.HEIGHT,

        road_mask,
        off_track_mask
    )


    cars.append(
        car
    )


    checkpoint_trackers.append(
        CheckpointTracker(
            CHECKPOINTS,
            lap_gates=LAP_LINES
        )
    )


    episode_steps.append(
        0
    )


    active.append(
        True
    )


# ============================================================
# CREATE MODELS
# ============================================================

models = []


if config.parent_evolution:

    # --------------------------------------------------------
    # ONE BRAIN PER CAR
    # --------------------------------------------------------

    for _ in range(
        config.NUM_CARS
    ):

        models.append(
            CarBrain()
        )


    model = None

    target_model = None

    optimizer = None


else:

    # --------------------------------------------------------
    # ONE SHARED DQN BRAIN
    # --------------------------------------------------------

    model = CarBrain()

    target_model = CarBrain()


    target_model.eval()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )


# ============================================================
# TRAINING SETTINGS
# ============================================================

episode = 0

step_count = 0

loss = None


TARGET_UPDATE = 500


if config.parent_evolution:

    epsilon = 0.0

    epsilon_min = 0.0

    epsilon_decay = 0.0


else:

    epsilon = 1.0

    epsilon_min = 0.05

    epsilon_decay = 0.995


# ============================================================
# LOAD SAVED MODEL
# ============================================================

if os.path.exists(
    MODEL_PATH
):

    # --------------------------------------------------------
    # EVOLUTION MODE
    #
    # Load the saved best model as one seed.
    # Leave the rest random to preserve diversity.
    # --------------------------------------------------------

    if config.parent_evolution:

        epsilon, episode = load_model(
            models[0]
        )


        print(
            "Loaded saved model as "
            "evolution seed."
        )


    # --------------------------------------------------------
    # DQN MODE
    # --------------------------------------------------------

    else:

        epsilon, episode = load_model(
            model,
            optimizer=optimizer
        )


        target_model.load_state_dict(
            model.state_dict()
        )


else:

    print(
        f"No saved model at "
        f"{MODEL_PATH} - "
        "starting from scratch."
    )


    if not config.parent_evolution:

        target_model.load_state_dict(
            model.state_dict()
        )


# ============================================================
# FITNESS
#
# ONE SCORE PER MODEL/CAR
# ============================================================

fitness_scores = [
    0.0
] * config.NUM_CARS


# ============================================================
# HUD
# ============================================================

def draw_hud(
    screen,
    episode,
    epsilon,
    loss,
    best_fitness
):

    alive_count = sum(
        active
    )


    best_checkpoints = max(
        (
            tracker.scored
            for tracker
            in checkpoint_trackers
        ),
        default=0
    )


    lines = [

        "Model: {}".format(
            config.MODEL_NAME
        ),

        "Mode: {}".format(
            "Evolution"
            if config.parent_evolution
            else "DQN"
        ),

        "cars        {}".format(
            config.NUM_CARS
        ),

        "alive       {}".format(
            alive_count
        ),

        "dead        {}".format(
            config.NUM_CARS
            -
            alive_count
        ),

        "best checkpoints {}".format(
            best_checkpoints
        ),

        "best fitness {:.2f}".format(
            best_fitness
        ),

        "episode     {}".format(
            episode
        )
    ]


    # --------------------------------------------------------
    # DQN-SPECIFIC HUD
    # --------------------------------------------------------

    if not config.parent_evolution:

        lines.extend(
            [
                "epsilon     {:.3f}".format(
                    epsilon
                ),

                "loss        {}".format(
                    "waiting"
                    if loss is None
                    else "{:.5f}".format(
                        loss
                    )
                )
            ]
        )


    for i, text in enumerate(
        lines
    ):

        readout = small_font.render(
            text,
            True,
            (255, 255, 255)
        )


        screen.blit(
            readout,
            (
                12,
                12 + i * 18
            )
        )

# ============================================================
# MAIN LOOP
# ============================================================

current_episode_count = 0
running = True
survivor_timer = 0


while running:
    # ========================================================
    # EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


        elif event.type == pygame.KEYDOWN:

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            if event.key == pygame.K_s:

                if config.parent_evolution:

                    best_index = max(
                        range(
                            config.NUM_CARS
                        ),
                        key=lambda index:
                            fitness_scores[index]
                    )


                    save_model(
                        models[
                            best_index
                        ],
                        episode
                    )


                else:

                    save_model(
                        model,
                        episode,
                        optimizer,
                        epsilon
                    )


            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            elif event.key == pygame.K_ESCAPE:

                running = False


    # ========================================================
    # UPDATE ACTIVE CARS
    # ========================================================
    count = 0
    for i, car in enumerate(
        cars
    ):

        # ----------------------------------------------------
        # DEAD CARS WAIT UNTIL GENERATION RESET
        # ----------------------------------------------------

        if not active[i]:

            continue


        # ====================================================
        # STATE
        # ====================================================

        state = car.get_state()


        # ====================================================
        # ACTION
        # ====================================================

        if config.parent_evolution:

            action = choose_action(
                models[i],
                state,
                epsilon=0.0
            )


        else:

            action = choose_action(
                model,
                state,
                epsilon
            )


        throttle, steering = (
            action_to_controls(
                action
            )
        )


        # ====================================================
        # POSITION BEFORE MOVEMENT
        # ====================================================

        previous_position = (
            car.x,
            car.y
        )


        # ====================================================
        # MOVE
        # ====================================================

        reward = car.update(
            throttle,
            steering
        )


        # ====================================================
        # CHECKPOINT REWARD
        # ====================================================

        reward += (
            checkpoint_trackers[
                i
            ].update(
                previous_position,
                (
                    car.x,
                    car.y
                )
            )
        )


        # ====================================================
        # FITNESS
        #
        # CRITICAL FIX:
        # Every car accumulates its own total reward.
        # ====================================================

        fitness_scores[
            i
        ] += reward


        # ====================================================
        # NEXT STATE
        # ====================================================

        next_state = car.get_state()


        # ====================================================
        # STEP COUNT
        # ====================================================

        episode_steps[
            i
        ] += 1


        # ====================================================
        # DONE
        # ====================================================

        done = (
            car.crashed
            or
            episode_steps[i]
            >= config.MAX_EPISODE_STEPS
        )


        # ====================================================
        # DQN EXPERIENCE
        #
        # Evolution does NOT need replay memory.
        # ====================================================

        if not config.parent_evolution:

            remember(
                state,
                action,
                reward,
                next_state,
                done
            )


        # ====================================================
        # MARK DEAD
        # ====================================================

        if done:

            active[i] = False


    # ========================================================
    # DQN TRAINING ONLY
    # ========================================================

    if not config.parent_evolution:

        loss = train_model(
            model,
            target_model,
            optimizer
        )


        step_count += 1


        if (
            step_count
            % TARGET_UPDATE
            == 0
        ):

            target_model.load_state_dict(
                model.state_dict()
            )


    # ========================================================
    # ALL CARS DEAD?
    # ========================================================

    if sum(active) <= (config.SURVIVOR_LIMIT):
        survivor_timer += 1
        if survivor_timer >= config.SURVIVOR_EXTRA_STEPS:
            for i in range(config.NUM_CARS):
                active[i] = False
    else:
        survivor_timer = 0

    
    all_dead = not any(
        active
    )

    # ========================================================
    # END GENERATION / EPISODE
    # ========================================================
    if all_dead:
        episode += 1
        current_episode_count += 1


        # ====================================================
        # RANK CARS BY WHOLE-RUN FITNESS
        # ====================================================

        ranking = sorted(
            range(config.NUM_CARS),
            key=lambda index:
                fitness_scores[index],
            reverse=True
        )


        best_index = ranking[0]

        second_index = ranking[1]

        third_index = ranking[2]


        best_fitness = fitness_scores[
            best_index
        ]


        second_fitness = fitness_scores[
            second_index
        ]


        third_fitness = fitness_scores[
            third_index
        ]


        average_fitness = (
            sum(
                fitness_scores
            )
            /
            len(
                fitness_scores
            )
        )


        best_checkpoints = max(
            (
                tracker.scored
                for tracker
                in checkpoint_trackers
            ),
            default=0
        )


        longest_run = max(
            episode_steps,
            default=0
        )


        # ====================================================
        # PARENT EVOLUTION
        # ====================================================

        if config.parent_evolution:

            parent1 = models[
                best_index
            ]

            parent2 = models[
                second_index
            ]

            parent3 = models[
                third_index
            ]


            # ------------------------------------------------
            # NEW GENERATION
            # ------------------------------------------------

            new_models = []


            # ------------------------------------------------
            # ELITISM
            #
            # Preserve top 3 exactly.
            # ------------------------------------------------

            new_models.append(
                parent1
            )

            new_models.append(
                parent2
            )

            new_models.append(
                parent3
            )


            # ------------------------------------------------
            # CREATE CHILDREN
            # ------------------------------------------------

            while (
                len(new_models)
                <
                config.NUM_CARS
            ):

                child = crossover(
                    parent1,
                    parent2,
                    parent3
                )


                new_models.append(
                    child
                )


            # ------------------------------------------------
            # REPLACE OLD POPULATION
            # ------------------------------------------------

            models = new_models


        # ====================================================
        # DQN EPSILON DECAY
        # ====================================================

        else:

            epsilon *= (
                epsilon_decay
            )


            epsilon = max(
                epsilon_min,
                epsilon
            )


        # ====================================================
        # PRINT GENERATION RESULTS
        # ====================================================

        print(
            "\nEpisode:",
            episode,

            "| Best fitness:",
            round(
                best_fitness,
                2
            ),

            "| Second:",
            round(
                second_fitness,
                2
            ),

            "| Third:",
            round(
                third_fitness,
                2
            ),

            "| Average:",
            round(
                average_fitness,
                2
            ),

            "| Best checkpoints:",
            best_checkpoints,

            "| Longest steps:",
            longest_run,

            "| survivor limit:", config.SURVIVOR_LIMIT,

            "| session episode:", current_episode_count
        )


        # ====================================================
        # AUTOSAVE
        # ====================================================

        if (
            SAVE_EVERY_EPISODES
            and
            episode
            % SAVE_EVERY_EPISODES
            == 0
        ):

            if config.parent_evolution:

                save_model(
                    models[0],
                    episode
                )


            else:

                save_model(
                    model,
                    episode,
                    optimizer,
                    epsilon
                )


        # ====================================================
        # RESET ALL CARS
        # ====================================================

        for i, car in enumerate(
            cars
        ):

            car.reset()


            checkpoint_trackers[
                i
            ].reset()


            episode_steps[
                i
            ] = 0


            active[i] = True


            fitness_scores[
                i
            ] = 0.0


        # Replay memory is irrelevant to evolution
        if config.parent_evolution:

            clear_memory()


    # ========================================================
    # CURRENT BEST FITNESS
    # ========================================================

    current_best_fitness = max(
        fitness_scores,
        default=0.0
    )


    # ========================================================
    # DRAW TRACK
    # ========================================================

    screen.blit(
        track_image,
        (0, 0)
    )


    # ========================================================
    # DRAW CARS
    # ========================================================

    for car in cars:

        car.draw(
            screen
        )


    # ========================================================
    # HUD
    # ========================================================

    draw_hud(
        screen,
        episode,
        epsilon,
        loss,
        current_best_fitness
    )


    pygame.display.flip()


    # ========================================================
    # FPS
    # ========================================================

    clock.tick(
        60
    )


# ============================================================
# SAVE BEFORE EXIT
# ============================================================

if config.parent_evolution:

    best_index = max(
        range(
            config.NUM_CARS
        ),
        key=lambda index:
            fitness_scores[index]
    )


    save_model(
        models[
            best_index
        ],
        episode
    )


else:

    save_model(
        model,
        episode,
        optimizer,
        epsilon
    )


pygame.quit()