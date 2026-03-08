import numpy as np


class AABBCollider:
    def __init__(self, size=None):
        """
        If size is provided it will be used directly.
        If size is None, it can later be computed from mesh vertices.
        """
        self.size = None if size is None else np.array(size, dtype=np.float32)
        # local offset of the collider center relative to object origin
        self.offset = np.zeros(3, dtype=np.float32)

    def fit_to_vertices(self, vertices):
        """
        Automatically compute collider size from mesh vertices.

        vertices: numpy array shape (N,3)
        """
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)

        # collider size
        self.size = (vmax - vmin).astype(np.float32)

        # collider center relative to mesh origin
        center = (vmin + vmax) * 0.5
        self.offset = center.astype(np.float32)

    def get_bounds(self, transform):
        if self.size is None:
            raise ValueError("AABBCollider size is None. Call fit_to_vertices() or provide size.")

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

        # compute world-space center including offset
        world_center = transform.position + (R @ (self.offset * transform.scale))

        min_v = world_center - extents
        max_v = world_center + extents

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
