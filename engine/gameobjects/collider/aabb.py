import numpy as np


class AABBCollider:
    def __init__(self, size=None):
        """
        Docstring für __init__

        :param self: The object itself
        :param size: The size of the collider
        """
        self.size = np.array(size, dtype=np.float32)

    def get_bounds(self, transform):

        world_size = self.size * transform.scale
        half = world_size * 0.5

        # extract rotation only (remove scaling from model matrix)
        R = transform.matrix()[:3, :3].copy()

        # normalize columns to remove scale influence
        R[:, 0] /= np.linalg.norm(R[:, 0]) if np.linalg.norm(R[:, 0]) != 0 else 1.0
        R[:, 1] /= np.linalg.norm(R[:, 1]) if np.linalg.norm(R[:, 1]) != 0 else 1.0
        R[:, 2] /= np.linalg.norm(R[:, 2]) if np.linalg.norm(R[:, 2]) != 0 else 1.0

        # rotated extents
        extents = np.abs(R) @ half

        min_v = transform.position - extents
        max_v = transform.position + extents

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
