import numpy as np


class AABBCollider:
    def __init__(self, size=(1, 1, 1)):
        """
        Docstring für __init__

        :param self: The object itself
        :param size: The size of the collider
        """
        self.size = np.array(size, dtype=np.float32)

    def get_bounds(self, transform):
        """
        Docstring für get_bounds

        :param self: The object itself
        :param transform: The transform of the object
        """
        # World-space collider size = visual scale × local collider size
        world_size = self.size * transform.scale

        # Small safety margin to prevent side-clipping (in meters)
        collision_margin = 0.4

        half = (world_size * 0.5) + collision_margin

        min_v = transform.position - half
        max_v = transform.position + half
        return min_v, max_v

    def get_corners(self, transform):
        """
        Returns the 8 world-space corner points of the AABB.
        Useful for debug gizmo rendering (wireframe).
        """
        min_v, max_v = self.get_bounds(transform)

        x0, y0, z0 = min_v
        x1, y1, z1 = max_v

        # fmt: off
        return np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float32)
        # fmt: on
