import numpy as np
import random as rm

class Cloth:
    """
    2D cloth grid using Position Based Dynamics (PBD).
    Debug-render friendly (provides line segments for gizmo rendering).
    """

    def __init__(
        self,
        origin=(0, 5, 0),
        width=4.0,
        height=4.0,
        segments_x=10,
        segments_y=10,
        gravity=(0, -9.81, 0),
        wind_strength=1.0
    ):
        self.origin = np.array(origin, dtype=np.float32)
        self.width = width
        self.height = height
        self.segments_x = segments_x
        self.segments_y = segments_y
        self.gravity = np.array(gravity, dtype=np.float32)
        self.wind_strength = wind_strength

        self.points = []
        self.prev_points = []
        self.fixed = []
        self.constraints = []

        self.time = 0.0

        # Per-point wind variation (phase + speed)
        self.wind_phase = []
        self.wind_speed = []

        self._build_grid()

    def _index(self, x, y):
        return y * (self.segments_x + 1) + x

    def _build_grid(self):
        # Create grid points
        for y in range(self.segments_y + 1):
            for x in range(self.segments_x + 1):
                px = self.origin[0] + (x / self.segments_x) * self.width
                py = self.origin[1] - (y / self.segments_y) * self.height
                pz = self.origin[2]

                pos = np.array([px, py, pz], dtype=np.float32)

                self.points.append(pos.copy())
                self.prev_points.append(pos.copy())

                # Randomized wind parameters per point
                self.wind_phase.append(rm.uniform(0.0, 6.28318))  # 0..2π
                self.wind_speed.append(rm.uniform(0.8, 3.0))

                # Fix top row (curtain rod)
                if y == 0:
                    self.fixed.append(True)
                else:
                    self.fixed.append(False)

        # Structural constraints (horizontal + vertical)
        for y in range(self.segments_y + 1):
            for x in range(self.segments_x + 1):
                if x < self.segments_x:
                    self._add_constraint(x, y, x + 1, y)
                if y < self.segments_y:
                    self._add_constraint(x, y, x, y + 1)

    def _add_constraint(self, x1, y1, x2, y2):
        i1 = self._index(x1, y1)
        i2 = self._index(x2, y2)

        p1 = self.points[i1]
        p2 = self.points[i2]

        rest_length = np.linalg.norm(p2 - p1)
        self.constraints.append((i1, i2, rest_length))

    def step(self, dt, iterations=1):
        damping = 0.995

        self.time += dt

        # Verlet integration with damping
        for i in range(len(self.points)):
            if self.fixed[i]:
                continue

            current = self.points[i]
            previous = self.prev_points[i]

            velocity = (current - previous) * damping
            self.prev_points[i] = current.copy()

            # Procedural wind (subsection variation)
            x = current[0]
            y = current[1]

            phase = self.wind_phase[i]
            speed = self.wind_speed[i]

            t = self.time * speed

            wx = np.sin(t * 2.0 + y * 1.5 + phase) * 0.5 * self.wind_strength
            wz = np.cos(t * 1.5 + x * 1.2 + phase) * 0.5 * self.wind_strength

            self.points[i][0] = current[0] + velocity[0] + (self.gravity[0] + wx) * dt * dt
            self.points[i][1] = current[1] + velocity[1] + self.gravity[1] * dt * dt
            self.points[i][2] = current[2] + velocity[2] + (self.gravity[2] + wz) * dt * dt

        # Solve constraints (optimized distance computation)
        for _ in range(iterations):
            for i1, i2, rest_length in self.constraints:
                p1 = self.points[i1]
                p2 = self.points[i2]

                delta = p2 - p1
                dist_sq = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]

                if dist_sq == 0.0:
                    continue

                dist = dist_sq ** 0.5
                diff = (dist - rest_length) / dist
                correction = delta * 0.5 * diff

                if not self.fixed[i1]:
                    self.points[i1] += correction
                if not self.fixed[i2]:
                    self.points[i2] -= correction
                    
    def build_mesh_data(self):
        """
        Generate mesh data (positions, normals, uvs, indices)
        for rendering the cloth.
        """
        vertices = np.array(self.points, dtype=np.float32)

        # UVs
        uvs = []
        for y in range(self.segments_y + 1):
            for x in range(self.segments_x + 1):
                u = x / self.segments_x
                v = y / self.segments_y
                uvs.append([u, v])

        uvs = np.array(uvs, dtype=np.float32)

        # Indices
        indices = []
        for y in range(self.segments_y):
            for x in range(self.segments_x):
                i0 = y * (self.segments_x + 1) + x
                i1 = i0 + 1
                i2 = i0 + (self.segments_x + 1)
                i3 = i2 + 1

                indices += [i0, i2, i1]
                indices += [i1, i2, i3]

        indices = np.array(indices, dtype=np.uint32)

        normals = np.zeros_like(vertices)

        for i in range(0, len(indices), 3):
            i0 = indices[i]
            i1 = indices[i + 1]
            i2 = indices[i + 2]

            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]

            edge1 = v1 - v0
            edge2 = v2 - v0

            face_normal = np.cross(edge1, edge2)

            normals[i0] += face_normal
            normals[i1] += face_normal
            normals[i2] += face_normal

        # Normalize
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1.0
        normals /= lengths[:, np.newaxis]

        return vertices, normals, uvs, indices

    def get_debug_lines(self):
        """
        Returns Nx2 line vertices for gizmo rendering.
        """
        lines = []

        for i1, i2, _ in self.constraints:
            lines.append(self.points[i1])
            lines.append(self.points[i2])

        return np.array(lines, dtype=np.float32)