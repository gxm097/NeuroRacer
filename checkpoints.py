"""
Checkpoint settings and detection.

Checkpoints are not hard-coded any more. You draw them straight
onto track.png as red lines, and this file finds them.

Draw the start/finish line in BLUE and crossing it completes a
lap, which is worth more than a checkpoint.

Each separate blob of red becomes one checkpoint. A line is
reduced to the straight segment running down its long axis, so
the thickness of your brush does not matter - only where the
line sits and which way it points.

Rules for drawing:

  * Draw each checkpoint as its own line. Two lines that touch
    are read as a single checkpoint.
  * Make the line span the full width of the road, otherwise the
    car can drive around the end of it and never score.
  * A little overspill onto the grass is fine and is expected.
  * The same applies to the blue start/finish line.

Detection runs on the track image, never on the rendered window,
so the red car sprite can never be mistaken for a checkpoint.
"""

import numpy as np
import config


# ============================================================
# DETECTION SETTINGS
# ============================================================

# A pixel counts as red when the red channel is at least this
# strong...
RED_MIN_CHANNEL = 100

# ...and beats both other channels by at least this much. The
# margin is what separates a drawn red line from the brownish
# and grey tones already present in a track image.
RED_DOMINANCE = 60


# A pixel counts as blue when the blue channel is strong and
# beats both other channels, mirroring the red test above.
BLUE_MIN_CHANNEL = 100

BLUE_DOMINANCE = 60


# Blobs smaller than this are ignored, so stray specks and the
# soft antialiased edge of a brush stroke do not turn into
# phantom checkpoints.
MIN_CHECKPOINT_PIXELS = 40


# ============================================================
# DRIVING SETTINGS
# ============================================================

# The coloured lines sit on top of the road, so without this the
# car would treat every checkpoint and the finish line as a wall
# and crash into them.
TREAT_RED_AS_ROAD = True

# How far past the edge of the real road a red pixel may still
# count as drivable. This allows overspill onto the grass while
# stopping a badly overdrawn line from becoming a shortcut
# across the infield. Applies to blue as well as red.
RED_ROAD_MARGIN = 6


# ============================================================
# SCORING SETTINGS
# ============================================================


# How many DIFFERENT checkpoints must be crossed before the same
# one can pay out again.
#
#   1 = must visit one other checkpoint before coming back.
#       This is the rule as asked for.
#
#   2+ = stricter. Raise this if the car learns to farm reward
#        by shuttling back and forth between two neighbouring
#        checkpoints, which a value of 1 still allows.
COOLDOWN_CHECKPOINTS = 1


# How many checkpoints must be collected before the blue line
# will count as a lap.
#
# Without this the car could park on the finish line and take
# LAP_REWARD over and over. Set it to the number of red lines
# you have drawn to force a full circuit; set it to 0 to let
# every crossing count.
#
# It is automatically capped at the number of checkpoints that
# actually exist, so drawing a blue line and no red ones still
# scores laps rather than silently never paying out.
LAP_REQUIRES_CHECKPOINTS = 0


# ============================================================
# RED DETECTION
# ============================================================

def red_mask(rgb):
    """
    rgb: (width, height, 3) array as returned by
         pygame.surfarray.array3d - note x first, then y.

    Returns a (width, height) boolean array of red pixels.
    """

    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)

    return (
        (r >= RED_MIN_CHANNEL)
        & ((r - g) >= RED_DOMINANCE)
        & ((r - b) >= RED_DOMINANCE)
    )


def blue_mask(rgb):
    """
    Same idea as red_mask, for the start/finish line.
    """

    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)

    return (
        (b >= BLUE_MIN_CHANNEL)
        & ((b - r) >= BLUE_DOMINANCE)
        & ((b - g) >= BLUE_DOMINANCE)
    )


def marker_mask(rgb):
    """
    Every drawn line, red or blue. main.py uses this to make the
    lines drivable so the car does not crash on them.
    """

    return red_mask(rgb) | blue_mask(rgb)


# ============================================================
# GROUP MARKED PIXELS INTO SEPARATE LINES
# ============================================================

def _blobs(mask):
    """
    Splits a boolean mask into connected blobs.

    Uses 8-way connectivity so a diagonal brush stroke stays in
    one piece instead of breaking up into a dotted line.
    """

    width, height = mask.shape

    seen = np.zeros_like(mask)

    xs, ys = np.where(mask)

    found = []

    for i in range(len(xs)):

        start_x = int(xs[i])
        start_y = int(ys[i])

        if seen[start_x, start_y]:
            continue

        stack = [(start_x, start_y)]
        seen[start_x, start_y] = True

        points = []

        while stack:

            x, y = stack.pop()
            points.append((x, y))

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):

                    nx = x + dx
                    ny = y + dy

                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and mask[nx, ny]
                        and not seen[nx, ny]
                    ):
                        seen[nx, ny] = True
                        stack.append((nx, ny))

        found.append(points)

    return found


def _blob_to_segment(points):
    """
    Reduces a blob of pixels to the straight line running along
    its longest axis.

    The direction comes from an SVD of the points, so it works
    at any angle, and the endpoints are the extreme points
    projected onto that direction. A thick brush and a thin one
    therefore produce the same segment.
    """

    pts = np.asarray(points, dtype=float)

    centre = pts.mean(axis=0)

    centred = pts - centre

    # Principal axis of the stroke.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)

    axis = vt[0]

    projection = centred @ axis

    p1 = centre + axis * projection.min()
    p2 = centre + axis * projection.max()

    return (
        (float(p1[0]), float(p1[1])),
        (float(p2[0]), float(p2[1]))
    )


def _lines_from_mask(mask):

    lines = []

    for points in _blobs(mask):

        if len(points) < MIN_CHECKPOINT_PIXELS:
            continue

        lines.append(_blob_to_segment(points))

    return lines


def find_checkpoints(rgb):
    """
    Every red line, as ((x1, y1), (x2, y2)) segments.

    The order of the list is not meaningful - scoring never
    assumes one.
    """

    return _lines_from_mask(red_mask(rgb))


def find_lap_lines(rgb):
    """
    Every blue line. Normally there is just one, but more than
    one is allowed and any of them completes a lap.
    """

    return _lines_from_mask(blue_mask(rgb))


# ============================================================
# GEOMETRY
# ============================================================

def segments_cross(a1, a2, b1, b2):
    """
    True when segment a1-a2 crosses segment b1-b2.
    """

    def side(o, p, q):
        return (
            (p[0] - o[0]) * (q[1] - o[1])
            - (p[1] - o[1]) * (q[0] - o[0])
        )

    d1 = side(b1, b2, a1)
    d2 = side(b1, b2, a2)
    d3 = side(a1, a2, b1)
    d4 = side(a1, a2, b2)

    return (
        ((d1 > 0) != (d2 > 0))
        and ((d3 > 0) != (d4 > 0))
    )


# ============================================================
# SCORING
# ============================================================

class CheckpointTracker:
    """
    Pays out reward for driving through the red lines.

    A checkpoint will not pay out twice in a row. Once it scores
    it goes on hold until COOLDOWN_CHECKPOINTS other checkpoints
    have been crossed, so the car cannot park on one line and
    farm it.
    """

    def __init__(
        self,
        gates,
        lap_gates=(),
        cooldown=COOLDOWN_CHECKPOINTS,
        reward=config.CHECKPOINT_REWARD,
        lap_reward=config.LAP_REWARD,
        lap_requires=LAP_REQUIRES_CHECKPOINTS
    ):

        self.gates = gates
        self.lap_gates = list(lap_gates)

        self.cooldown = cooldown
        self.reward = reward
        self.lap_reward = lap_reward

        # Never demand more checkpoints than exist, otherwise a
        # track with a blue line and no red ones could never
        # complete a lap.
        self.lap_requires = min(
            lap_requires,
            len(self.gates)
        )

        self.reset()


    def reset(self):

        # Most recently scored checkpoints, newest first.
        self.recent = []

        self.scored = 0
        self.last_index = None

        self.laps = 0

        # Counts toward the next lap, then resets.
        self.checkpoints_since_lap = 0


    def update(self, previous_position, new_position):
        """
        Call once per frame with where the car was and where it
        now is. Returns the reward earned this frame.

        Within a single frame the car travels in a straight line
        because its heading is fixed for the whole of
        update_position, so testing that one segment against
        each gate catches the crossing exactly. There is no way
        to skip through a line by moving quickly.
        """

        if not self.gates and not self.lap_gates:
            return 0.0

        earned = 0.0

        for index, (gate_p, gate_q) in enumerate(self.gates):

            if not segments_cross(
                previous_position,
                new_position,
                gate_p,
                gate_q
            ):
                continue


            # Still on hold from the last time it scored.
            if index in self.recent[:self.cooldown]:
                continue


            earned += self.reward
            self.scored += 1

            if index in self.recent:
                self.recent.remove(index)

            self.recent.insert(0, index)

            del self.recent[self.cooldown + 2:]

            self.last_index = index

            self.checkpoints_since_lap += 1


        # ----------------------------------------------------
        # Lap line
        # ----------------------------------------------------

        for gate_p, gate_q in self.lap_gates:

            if not segments_cross(
                previous_position,
                new_position,
                gate_p,
                gate_q
            ):
                continue


            # Not enough of the track covered yet, so this is
            # the car re-crossing the line rather than finishing
            # a lap.
            if self.checkpoints_since_lap < self.lap_requires:
                continue


            earned += self.lap_reward

            self.laps += 1

            self.checkpoints_since_lap = 0

        return earned
