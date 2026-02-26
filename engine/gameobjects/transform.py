import numpy as np

class Transform:
    def __init__(self, position=(0, 0, 0), scale=(1, 1, 1), yaw=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.yaw = yaw

    def matrix(self):
        m = np.identity(4, dtype=np.float32)

        # Rotation aus yaw (NICHT auf bestehende Matrix!)
        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        rot = np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.float32)

        # Scale
        scale = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])

        # Combine cleanly
        m = rot @ scale
        m[:3, 3] = self.position
        return m