import numpy as np
import math

from .player import look_at


class Camera:
    def __init__(self, player, physics_world, fov=120.0):
        self.player = player
        self.physics_world = physics_world
        self.fov = fov

        # Eye height as a fraction of player height (single source of truth)
        self.eye_height_factor = 0.4

        # First-person settings
        self.fps_forward_offset = 0.45
        self.fps_eye_vertical_bias = 0.30

        # Third-person settings
        self.third_person = False
        self.distance = 4.0
        self.height_offset = 1.2
        self.collision_padding = 0.5  # distance kept from walls

    def get_view_matrix(self) -> np.ndarray:
        if not self.third_person:
            # -------- First Person --------
            eye = self.player.position.copy()

            # move from capsule center to eye height
            eye[1] += self.player.height * self.eye_height_factor
            eye[1] += self.player._headbob_offset
            eye[1] += self.fps_eye_vertical_bias

            # offset prevents clipping through mannequin head
            eye += self.player.front * self.fps_forward_offset

            return look_at(eye, eye + self.player.front, self.player.up)

        else:
            # -------- Third Person (collision aware) --------
            target = self.player.position.copy()
            target[1] += self.player.height * self.eye_height_factor

            # desired camera position behind the player
            desired_eye = (
                target
                - self.player.front * self.distance
                + self.player.up * self.height_offset
            )

            # camera collision via raycast
            hit = None
            if self.physics_world is not None:
                hit = self.physics_world.raycast(target, desired_eye)

            if hit is not None:
                # pull camera closer to avoid clipping into geometry
                eye = hit + self.player.front * (self.collision_padding + 0.1)
            else:
                eye = desired_eye

            return look_at(eye, target, self.player.up)

    def get_projection_matrix(self, aspect: float, near=0.1, far=100.0) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0

        return proj
