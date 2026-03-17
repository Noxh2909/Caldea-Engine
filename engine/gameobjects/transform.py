import numpy as np


class Transform:
    def __init__(self, position=(0, 0, 0), scale=(1, 1, 1), yaw=0.0, roll=0.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.position[1] += (
            self.scale[1] * 0.5
        )  # Adjust Y position to account for scale (assuming pivot at bottom)

        # Accept yaw in degrees; convert to radians internally
        self.yaw = 0.0 if yaw is None else np.radians(yaw)
        self.roll = 0.0 if roll is None else np.radians(roll)
        self.pitch = 0.0 if pitch is None else np.radians(pitch)
        
    def matrix(self):
        m = np.identity(4, dtype=np.float32)

        # Yaw rotation (Y-axis)
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)

        rot_yaw = np.array([
            [ cy, 0, sy, 0],
            [  0, 1,  0, 0],
            [-sy, 0, cy, 0],
            [  0, 0,  0, 1],
        ], dtype=np.float32)

        # Roll rotation (Z-axis)
        cr = np.cos(self.roll)
        sr = np.sin(self.roll)

        rot_roll = np.array([
            [cr, -sr, 0, 0],
            [sr,  cr, 0, 0],
            [ 0,   0, 1, 0],
            [ 0,   0, 0, 1],
        ], dtype=np.float32)

        # Pitch rotation (X-axis)
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)

        rot_pitch = np.array([
            [1,  0,   0, 0],
            [0, cp, -sp, 0],
            [0, sp,  cp, 0],
            [0,  0,   0, 1],
        ], dtype=np.float32)

        # Combine rotations
        rot = rot_yaw @ rot_roll @ rot_pitch

        # Scale
        scale = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])

        # Combine cleanly
        m = rot @ scale
        m[:3, 3] = self.position
        return m
