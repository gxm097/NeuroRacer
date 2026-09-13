"""
Render a demo GIF and screenshot of one evolution generation, headless.

    python tools/render_demo.py                      # defaults from config.py
    python tools/render_demo.py --track track.png --model simpleJoe.pth --seconds 15

Car 0 gets the saved model; every other car is a mutated child of it
(crossover with itself), so the GIF shows a realistic generation rather
than 100 identical drivers. Needs ffmpeg on PATH for the GIF step.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pygame
import torch

import config
from Brain import CarBrain
from car import Car
from checkpoints import (
    CheckpointTracker,
    RED_ROAD_MARGIN,
    TREAT_RED_AS_ROAD,
    find_checkpoints,
    find_lap_lines,
    marker_mask,
)
from evolutionHandle import crossover
from trainer import action_to_controls, choose_action


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--track", default=config.TRACK_NAME, help="image in tracks/")
parser.add_argument("--model", default=config.MODEL_NAME, help="checkpoint in models/ used as the seed")
parser.add_argument("--cars", type=int, default=config.NUM_CARS)
parser.add_argument("--seconds", type=float, default=12, help="max sim length at 60 fps; stops early if all cars crash")
parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")
parser.add_argument("--width", type=int, default=720, help="GIF width in px")
parser.add_argument("--shot-at", type=int, default=180, help="sim frame to save as the screenshot")
parser.add_argument("--gif", default=os.path.join(ROOT, "docs", "demo.gif"))
parser.add_argument("--screenshot", default=os.path.join(ROOT, "docs", "screenshot.png"))
parser.add_argument("--seed", type=int, default=7, help="torch seed for the mutations")
args = parser.parse_args()

torch.manual_seed(args.seed)
pygame.init()
screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
small_font = pygame.font.SysFont("consolas", 16)


# ============================================================
# TRACK + ROAD MASK (same rules as main.py)
# ============================================================

track_native = pygame.image.load(os.path.join(ROOT, "tracks", args.track)).convert()
track_image = pygame.transform.smoothscale(track_native, (config.WIDTH, config.HEIGHT))
collision_source = pygame.transform.scale(track_native, (config.WIDTH, config.HEIGHT))


def build_drivable_array(surface):
    rgb = pygame.surfarray.array3d(surface).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = rgb.mean(axis=2)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    gray = (spread <= 25) & (brightness >= 60) & (brightness <= 215)
    tan = (r > g) & (g > b) & ((r - b) >= 25) & ((r - b) <= 110) & (brightness >= 140)
    red = (r >= 150) & (r >= g + 80) & (r >= b + 80)
    blue = (b >= 150) & (b >= g + 60) & (b >= r + 60)
    return gray | tan | red | blue


def dilate(mask, radius):
    out = mask.copy()
    for shift in range(-radius, radius + 1):
        out |= np.roll(mask, shift, axis=0)
    wide = out.copy()
    for shift in range(-radius, radius + 1):
        out |= np.roll(wide, shift, axis=1)
    return out


drivable = build_drivable_array(collision_source)
rgb = pygame.surfarray.array3d(collision_source).astype(np.int16)
if TREAT_RED_AS_ROAD:
    drivable = drivable | (marker_mask(rgb) & dilate(drivable, RED_ROAD_MARGIN))

CHECKPOINTS = find_checkpoints(rgb)
LAP_LINES = find_lap_lines(rgb)
print(f"Found {len(CHECKPOINTS)} checkpoint(s), {len(LAP_LINES)} lap line(s).")

mask_rgb = np.repeat(drivable[:, :, None], 3, axis=2).astype(np.uint8) * 255
road_mask = pygame.mask.from_threshold(pygame.surfarray.make_surface(mask_rgb), (255, 255, 255), (50, 50, 50))
if not road_mask.get_at((config.START_X, config.START_Y)):
    sys.exit("start position is not on the road for this track - fix START_X/START_Y in config.py")
road_mask = road_mask.connected_component((config.START_X, config.START_Y))
off_track_mask = road_mask.copy()
off_track_mask.invert()


# ============================================================
# POPULATION: saved model + mutated children
# ============================================================

seed = CarBrain()
checkpoint = torch.load(os.path.join(ROOT, "models", args.model), weights_only=False)
seed.load_state_dict(checkpoint["model_state"])
print(f"Seeded from {args.model} (episode {checkpoint.get('episode', '?')})")

models = [seed] + [crossover(seed, seed, seed) for _ in range(args.cars - 1)]

cars, trackers, fitness, active = [], [], [], []
for _ in range(args.cars):
    cars.append(Car(config.START_X, config.START_Y, config.START_ANGLE,
                    config.WIDTH, config.HEIGHT, road_mask, off_track_mask))
    trackers.append(CheckpointTracker(CHECKPOINTS, lap_gates=LAP_LINES))
    fitness.append(0.0)
    active.append(True)


def draw(frame, lead):
    screen.blit(track_image, (0, 0))
    for i, car in enumerate(cars):
        if i == lead and not car.crashed:
            car.draw_sensors(screen)
        screen.blit(car.image, car.rect)
    lines = [
        f"Mode: Evolution   cars {args.cars}   alive {sum(active)}",
        f"best checkpoints {max(t.scored for t in trackers)}   best fitness {max(fitness):.1f}",
        f"step {frame}",
    ]
    for k, text in enumerate(lines):
        screen.blit(small_font.render(text, True, (255, 255, 255)), (12, 12 + k * 18))


# ============================================================
# SIMULATE + DUMP FRAMES
# ============================================================

every = max(1, round(60 / args.fps))
frames_dir = tempfile.mkdtemp(prefix="ml-car-frames-")
kept = 0

try:
    for frame in range(int(args.seconds * 60)):
        for i, car in enumerate(cars):
            if not active[i]:
                continue
            throttle, steering = action_to_controls(choose_action(models[i], car.get_state()))
            prev = (car.x, car.y)
            reward = car.update(throttle, steering)
            reward += trackers[i].update(prev, (car.x, car.y))
            fitness[i] += reward
            if car.crashed:
                active[i] = False

        lead = max(range(args.cars), key=lambda i: (active[i], fitness[i]))

        if frame % every == 0:
            draw(frame, lead)
            pygame.image.save(screen, os.path.join(frames_dir, f"frame_{kept:04d}.png"))
            kept += 1

        if frame == args.shot_at:
            draw(frame, lead)
            os.makedirs(os.path.dirname(args.screenshot), exist_ok=True)
            pygame.image.save(screen, args.screenshot)
            print(f"Saved {args.screenshot}")

        if not any(active):
            print(f"All cars crashed at frame {frame}.")
            break

    print(f"{kept} frames | best fitness {max(fitness):.1f} | best checkpoints {max(t.scored for t in trackers)}")

    if shutil.which("ffmpeg") is None:
        sys.exit(f"ffmpeg not found - frames are in {frames_dir}")

    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-i", os.path.join(frames_dir, "frame_%04d.png"),
            "-vf", f"scale={args.width}:-1:flags=lanczos,split[s0][s1];"
                   "[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop", "0",
            args.gif,
        ],
        check=True,
    )
    print(f"Saved {args.gif} ({os.path.getsize(args.gif) // 1024} KB)")

finally:
    shutil.rmtree(frames_dir, ignore_errors=True)
