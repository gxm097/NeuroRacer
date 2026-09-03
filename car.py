import pygame
import math
import config



class Car:
    def __init__(
        self,
        x,
        y,
        angle,
        width,
        height,
        road_mask,
        off_track_mask
    ):

        self.start_x = float(x)
        self.start_y = float(y)
        self.start_angle = float(angle)

        # Game environment
        self.width = width
        self.height = height

        self.road_mask = road_mask
        self.off_track_mask = off_track_mask

        # Car settings
        self.max_speed = config.MAX_SPEED
        self.reverse_speed = config.REVERSE_SPEED

        self.acceleration = config.ACCELERATION
        self.friction = config.FRICTION

        self.turn_speed = config.TURN_SPEED

        # Sensors
        self.sensor_length = config.SENSOR_LENGTH

        self.sensor_angles = config.SENSOR_ANGLES

        # Car image
        self.original_image = pygame.Surface(
            (22, 10),
            pygame.SRCALPHA
        )

        self.original_image.fill(
            (255, 135, 67)
        )

        self.reset()


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.x = self.start_x
        self.y = self.start_y

        self.angle = self.start_angle
        self.speed = 0.0

        self.crashed = False

        self.last_reward = 0.0

        self.sensor_endpoints = []

        self.sensor_readings = [
            self.sensor_length
        ] * len(self.sensor_angles)

        self.update_rect()
        self.update_sensors()


    # ========================================================
    # CONTROLS
    # ========================================================

    def apply_controls(
        self,
        throttle,
        steering
    ):

        if self.crashed:

            self.speed = 0.0
            return


        # Forward
        if throttle > 0:

            self.speed += self.acceleration


        # Reverse
        elif throttle < 0:

            self.speed -= self.acceleration


        # Friction
        else:

            if self.speed > 0:

                self.speed -= self.friction

                if self.speed < 0:
                    self.speed = 0


            elif self.speed < 0:

                self.speed += self.friction

                if self.speed > 0:
                    self.speed = 0


        # Limit speed
        self.speed = max(
            -self.reverse_speed,
            min(
                self.speed,
                self.max_speed
            )
        )


        # Steering
        if self.speed != 0:

            self.angle += (
                steering
                * self.turn_speed
            )


    # ========================================================
    # MOVEMENT
    # ========================================================

    def update_position(self):

        if self.crashed:
            return


        old_x = self.x
        old_y = self.y
        old_angle = self.angle


        radians = math.radians(
            self.angle
        )


        step_x = (
            math.cos(radians)
            * self.speed
        )

        step_y = (
            -math.sin(radians)
            * self.speed
        )


        distance = math.hypot(
            step_x,
            step_y
        )


        steps = max(
            1,
            int(math.ceil(distance))
        )


        for _ in range(steps):

            prev_x = self.x
            prev_y = self.y


            self.x += (
                step_x / steps
            )

            self.y += (
                step_y / steps
            )


            self.update_rect()


            # Car left track
            if self.is_off_track():

                self.x = old_x
                self.y = old_y
                self.angle = old_angle

                self.speed = 0.0
                self.crashed = True

                self.last_reward -= config.CRASH_PENALTY

                self.update_rect()

                return


            old_x = self.x
            old_y = self.y


    # ========================================================
    # ROTATED CAR
    # ========================================================

    def update_rect(self):

        self.image = (
            pygame.transform.rotate(
                self.original_image,
                self.angle
            )
        )


        self.rect = (
            self.image.get_rect(
                center=(
                    int(self.x),
                    int(self.y)
                )
            )
        )


        self.mask = (
            pygame.mask.from_surface(
                self.image
            )
        )


    # ========================================================
    # HITBOX
    # ========================================================

    def is_off_track(self):

        # Outside window
        if (
            self.rect.left < 0
            or self.rect.right >= self.width
            or self.rect.top < 0
            or self.rect.bottom >= self.height
        ):

            return True


        # Car overlaps non-road pixels
        hit = (
            self.off_track_mask.overlap(
                self.mask,
                (
                    self.rect.left,
                    self.rect.top
                )
            )
        )


        return hit is not None


    # ========================================================
    # SENSOR
    # ========================================================

    def cast_sensor(
        self,
        relative_angle
    ):

        radians = math.radians(
            self.angle
            + relative_angle
        )


        cos_a = math.cos(radians)
        sin_a = math.sin(radians)


        for distance in range(
            self.sensor_length
        ):

            x = int(
                self.x
                + cos_a * distance
            )

            y = int(
                self.y
                - sin_a * distance
            )


            # Outside screen
            if (
                x < 0
                or x >= self.width
                or y < 0
                or y >= self.height
            ):

                return (
                    x,
                    y,
                    distance
                )


            # Sensor sees edge
            if (
                self.road_mask.get_at(
                    (x, y)
                )
                == 0
            ):

                return (
                    x,
                    y,
                    distance
                )


        end_x = int(
            self.x
            + cos_a
            * self.sensor_length
        )

        end_y = int(
            self.y
            - sin_a
            * self.sensor_length
        )


        return (
            end_x,
            end_y,
            self.sensor_length
        )


    # ========================================================
    # SENSOR UPDATE
    # ========================================================

    def update_sensors(self):

        if self.crashed:
            return


        self.sensor_readings = []

        self.sensor_endpoints = []


        for sensor_angle in (
            self.sensor_angles
        ):

            (
                end_x,
                end_y,
                distance
            ) = self.cast_sensor(
                sensor_angle
            )


            self.sensor_readings.append(
                distance
            )

            self.sensor_endpoints.append(
                (end_x, end_y)
            )


    # ========================================================
    # ML STATE
    # ========================================================

    def get_state(self):

        normalized_sensors = [

            distance
            / self.sensor_length

            for distance
            in self.sensor_readings
        ]


        normalized_speed = (
            self.speed
            / self.max_speed
        )


        return (
            normalized_sensors

            + [
                normalized_speed
            ]

            + [
                1.0
                if self.crashed
                else 0.0
            ]
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        throttle,
        steering
    ):

        self.last_reward = 0.0


        self.apply_controls(
            throttle,
            steering
        )


        self.update_position()

        self.update_sensors()


        return self.last_reward


    # ========================================================
    # DRAW
    # ========================================================

    def draw_sensors(
        self,
        screen
    ):

        for end_point in (
            self.sensor_endpoints
        ):

            pygame.draw.line(
                screen,
                (160, 180, 255),

                (
                    int(self.x),
                    int(self.y)
                ),

                end_point,

                1
            )


            pygame.draw.circle(
                screen,
                (200, 210, 255),

                end_point,

                2
            )


    def draw(
        self,
        screen
    ):

        if config.show_sens:
            if not self.crashed:

                self.draw_sensors(
                    screen            #sensors show
                )


        screen.blit(
            self.image,
            self.rect
        )