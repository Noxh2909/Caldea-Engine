import numpy as np
import math


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Docstring für normalize

    :param v: The vector to normalize
    :type v: np.ndarray
    :return: The normalized vector
    :rtype: ndarray[Any, Any]
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def look_at(eye, center, up) -> np.ndarray:
    """
    Docstring für look_at

    :param eye: The camera position
    :param center: The point the camera is looking at
    :param up: The up vector
    :return: The view matrix
    :rtype: ndarray[Any, Any]
    """
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(s, eye)
    view[1, 3] = -np.dot(u, eye)
    view[2, 3] = np.dot(f, eye)

    return view


class Player:
    def __init__(
        self,
        position=(0.0, 4.0, 3.0),
        yaw=-90.0,
        pitch=0.0,
        speed=5.0,
        sensitivity=0.1,
    ):
        """
        Docstring für __init__

        :param self: The object itself
        :param position: The position of the camera
        :param yaw: The yaw of the camera
        :param pitch: The pitch of the camera
        :param speed: The speed of the camera
        :param sensitivity: The sensitivity of the mouse
        :param fov: The field of view of the camera
        """
        self.position = np.array(position, dtype=np.float32)
        # Previous frame position (set each frame by main.py for physics)
        self.prev_position = self.position.copy()
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.sensitivity = sensitivity

        # Minimal state only, physics controls these
        self.velocity_y = 0.0
        self.on_ground = False

        # Player hitbox
        self.radius = 0.35  # Breite des Spielers (X/Z)
        self.capsule_height = 2.0  # Höhe der Kapsel (Y)
        self.height = 2.0  # Gesamthöhe (Y)

        # Crouch
        self.stand_height = 1.6
        self.crouch_height = 0.5
        self.crouch_speed = 10.0  # how fast we interpolate between stand/crouch

        # Current interpolated player height (used by physics & camera)
        self.height = self.stand_height

        self.world_up = np.array((0.0, 1.0, 0.0), dtype=np.float32)

        self.front = np.zeros(3, dtype=np.float32)
        self.right = np.zeros(3, dtype=np.float32)
        self.up = np.zeros(3, dtype=np.float32)

        self._update_vectors()

        # Sprint
        self.walk_speed = speed
        self.sprint_speed = speed * 2.6

        # Slide
        self.is_sliding = False
        self.slide_velocity = np.zeros(3, dtype=np.float32)
        self.slide_friction = 6.0  # higher = stops faster
        self.slide_min_speed = 0.25  # below this, slide ends
        self.slide_speed_multiplier = 1.0  # scales sprint speed into slide speed

        # Air momentum
        self._air_velocity = np.zeros(3, dtype=np.float32)

        # Air control (small influence while airborne)
        self.air_control = 1.6  # 0.05–0.12 recommended

        # Jump (stronger takeoff, less moon-like)
        self.jump_strength = 12.0

        # Jump state
        self._jump_locked = False

        # Head bob
        self.headbob_time = 0.0
        self.headbob_amount = 0.12
        self.headbob_speed_walk = 8.0
        self.headbob_speed_sprint = 14.0
        self._headbob_offset = 0.0

    # =========================
    # Input Handling
    # =========================

    def process_mouse(self, dx: float, dy: float):
        """
        Docstring für process_mouse

        :param self: The object itself
        :param dx: The change in X position
        :type dx: float
        :param dy: The change in Y position
        :type dy: float
        """
        dx *= self.sensitivity
        dy *= self.sensitivity

        self.yaw += dx
        self.pitch -= dy  # invert Y for FPS feel

        self.pitch = max(-50.0, min(50.0, self.pitch))

        self._update_vectors()

    def process_keyboard(self, keys: dict, delta_time: float):
        """
        Docstring für process_keyboard

        :param self: The object itself
        :param keys: The keys pressed
        :type keys: dict
        :param delta_time: The time since the last frame
        :type delta_time: float
        """

        # -----------------
        # Sprint state (only selectable on ground)
        # -----------------
        is_sprinting = (
            keys.get("sprint", False) and self.on_ground and not self.is_sliding
        )
        # base movement speed
        current_speed = self.sprint_speed if is_sprinting else self.walk_speed

        # -----------------
        # Direction vectors (XZ)
        # -----------------
        forward = np.array([self.front[0], 0.0, self.front[2]], dtype=np.float32)
        forward = normalize(forward)

        right = np.array([self.right[0], 0.0, self.right[2]], dtype=np.float32)
        right = normalize(right)

        move_dir = np.zeros(3, dtype=np.float32)

        if keys.get("forward"):
            move_dir += forward
        if keys.get("backward"):
            move_dir -= forward
        if keys.get("left"):
            move_dir -= right
        if keys.get("right"):
            move_dir += right
        if np.linalg.norm(move_dir) > 0:
            move_dir = normalize(move_dir)
        # Crouch slows movement when grounded
        if keys.get("crouch_hold") and not self.is_sliding:
            current_speed *= 0.6

        # -----------------
        # Ground movement
        # -----------------
        if self.on_ground and not self.is_sliding:
            self._air_velocity = move_dir * current_speed
            self.position += self._air_velocity * delta_time

        # -----------------
        # Jump (impulse only)
        # -----------------
        if keys.get("jump") and self.on_ground and not self._jump_locked:
            self.velocity_y = self.jump_strength
            self.on_ground = False
            self._jump_locked = True

        # -----------------
        # Slide start (sprint + crouch on ground)
        # -----------------
        if (
            self.on_ground
            and not self.is_sliding
            and is_sprinting  # sprint required
            and keys.get("crouch_tap")  # edge-triggered tap
            and np.linalg.norm(self._air_velocity) > 0.0
        ):
            self.is_sliding = True

            # Slide speed derives from sprint speed
            slide_dir = normalize(self._air_velocity)
            self.slide_velocity = slide_dir * (
                self.sprint_speed * self.slide_speed_multiplier
            )

        # -----------------
        # Air movement (preserve momentum)
        # -----------------
        if not self.on_ground:
            # Apply small air control (no sprinting, no full steering)
            air_dir = np.zeros(3, dtype=np.float32)

            if keys.get("forward"):
                air_dir += forward
            if keys.get("backward"):
                air_dir -= forward
            if keys.get("left"):
                air_dir -= right
            if keys.get("right"):
                air_dir += right

            if np.linalg.norm(air_dir) > 0:
                air_dir = normalize(air_dir)
                self._air_velocity += (
                    air_dir * self.walk_speed * self.air_control * delta_time
                )

            self.position += self._air_velocity * delta_time

        # -----------------
        # Slide cancel conditions (input-driven)
        # -----------------
        if self.is_sliding:
            # Cancel slide on jump
            if keys.get("jump"):
                self.is_sliding = False
                self.slide_velocity[:] = 0.0

            # Cancel slide if sprint key is released
            elif not keys.get("sprint"):
                self.is_sliding = False
                self.slide_velocity[:] = 0.0

            # Cancel slide if player changes movement direction
            elif np.linalg.norm(move_dir) > 0:
                slide_dir = normalize(self.slide_velocity)
                if np.dot(slide_dir, move_dir) < 0.3:
                    self.is_sliding = False
                    self.slide_velocity[:] = 0.0

        # -----------------
        # Slide update (friction-based deceleration)
        # -----------------
        if self.is_sliding:
            self.position += self.slide_velocity * delta_time

            # Apply friction
            speed = np.linalg.norm(self.slide_velocity)
            speed -= self.slide_friction * delta_time
            speed = max(speed, 0.0)

            if speed <= self.slide_min_speed:
                self.is_sliding = False
                self.slide_velocity[:] = 0.0
            else:
                self.slide_velocity = normalize(self.slide_velocity) * speed

        # -----------------
        # Crouch / slide posture (FINAL, correct)
        # -----------------
        if self.is_sliding:
            target_height = self.crouch_height
            self.headbob_amount = 0.0

        elif keys.get("crouch_hold"):
            # crouch is purely a posture, not physics-dependent
            target_height = self.crouch_height
            self.headbob_amount = 0.0
        elif is_sprinting:
            target_height = self.stand_height
            self.headbob_amount = 0.08
        else:
            target_height = self.stand_height
            self.headbob_amount = 0.04

        # Smoothly interpolate player height
        self.height += (target_height - self.height) * min(
            1.0, self.crouch_speed * delta_time
        )

        # -----------------
        # Head bob (only when moving on ground)
        # -----------------
        if self.on_ground and np.linalg.norm(move_dir) > 0:
            bob_speed = (
                self.headbob_speed_sprint if is_sprinting else self.headbob_speed_walk
            )
            self.headbob_time += delta_time * bob_speed
            self._headbob_offset = np.sin(self.headbob_time) * self.headbob_amount
        else:
            self.headbob_time = 0.0
            self._headbob_offset = 0.0

    # =========================
    # Internal
    # =========================

    def _update_vectors(self):
        """
        Docstring für _update_vectors

        :param self: The object itself
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        front = np.array(
            [
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad),
            ],
            dtype=np.float32,
        )

        self.front = normalize(front)
        self.right = normalize(np.cross(self.front, self.world_up))
        self.up = normalize(np.cross(self.right, self.front))
