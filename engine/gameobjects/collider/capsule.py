import numpy as np


class CapsuleCollider:
    """
    Vertical capsule collider (Y-up).
    Used for player / characters.

    The capsule consists of:
    - a vertical cylinder
    - a hemisphere at the top
    - a hemisphere at the bottom
    """

    def __init__(self, radius: float = 0.35, height: float = 1.8):
        assert height >= 2 * radius, "Capsule height must be >= 2 * radius"

        self.radius = float(radius)
        self.height = float(height)

    # --------------------------------------------------
    # basic geometry
    # --------------------------------------------------

    @property
    def half_height(self) -> float:
        return self.height * 0.5

    @property
    def cylinder_height(self) -> float:
        return self.height - 2.0 * self.radius

    # --------------------------------------------------
    # world-space helpers
    # --------------------------------------------------

    def get_endpoints(self, position: np.ndarray):
        """
        Returns the centers of the bottom and top spheres.
        """
        bottom = position + np.array([0.0, self.radius, 0.0], dtype=np.float32)
        top = position + np.array([0.0, self.radius + self.cylinder_height, 0.0], dtype=np.float32)
        return bottom, top

    def get_aabb(self, position: np.ndarray):
        """
        Axis-aligned bounding box of the capsule.
        Used for broad-phase collision.
        """
        r = self.radius
        min_v = position + np.array([-r, 0.0, -r], dtype=np.float32)
        max_v = position + np.array([r, self.height, r], dtype=np.float32)
        return min_v, max_v

    # --------------------------------------------------
    # collision tests
    # --------------------------------------------------

    def intersects_aabb(self, position: np.ndarray, aabb_min, aabb_max) -> bool:
        """
        Capsule vs AABB intersection test.
        Used for player vs world geometry.
        """

        bottom, top = self.get_endpoints(position)

        # clamp aabb center to capsule segment
        closest = self._closest_point_on_segment(bottom, top, np.maximum(aabb_min, np.minimum((aabb_min + aabb_max) * 0.5, aabb_max)))

        # distance from closest point to AABB
        closest_aabb = np.maximum(aabb_min, np.minimum(closest, aabb_max))

        delta = closest - closest_aabb
        return np.dot(delta, delta) <= self.radius * self.radius

    # --------------------------------------------------
    # internal math
    # --------------------------------------------------

    @staticmethod
    def _closest_point_on_segment(a, b, p):
        """
        Closest point to p on segment ab.
        """
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        return a + ab * t
