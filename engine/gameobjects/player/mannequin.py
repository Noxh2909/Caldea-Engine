import math
import numpy as np
from typing import Optional
from gameobjects.player.animator import Animator


class Bone:
    def __init__(self, name: str, parent: int, inverse_bind: np.ndarray):
        self.name = name
        self.parent = parent         
        self.inverse_bind = inverse_bind.astype(np.float32)


class Skeleton:
    def __init__(self, bones: list['Bone']): 
        self.bones = bones
        self.parents = [b.parent for b in bones]

        # name → index lookup
        self.name_to_index = {
            b.name: i for i, b in enumerate(bones)
        }

        # runtime pose data
        self.local_matrices = [
            np.identity(4, dtype=np.float32) for _ in bones
        ]
        self.global_matrices = [
            np.identity(4, dtype=np.float32) for _ in bones
        ]

        # final matrices sent to GPU
        self.bone_matrices = np.repeat(
            np.identity(4, dtype=np.float32)[None, :, :],
            len(bones),
            axis=0
        )

    def set_rest_pose(self) -> None:
        """
        Initialize skeleton to rest pose (identity local transforms).
        """
        for i in range(len(self.bones)):
            self.local_matrices[i][:] = np.identity(4, dtype=np.float32)
            self.global_matrices[i][:] = np.identity(4, dtype=np.float32)
            self.bone_matrices[i][:] = self.bones[i].inverse_bind

    def update_from_local(self):
        """
        Propagate local → global → final bone matrices.
        Call once per frame after updating local_matrices.
        """
        for i, bone in enumerate(self.bones):
            if bone.parent >= 0:
                self.global_matrices[i][:] = self.global_matrices[bone.parent] @ self.local_matrices[i]
            else:
                self.global_matrices[i][:] = self.local_matrices[i]
            # bone.inverse_bind is stored per Bone
            self.bone_matrices[i][:] = self.global_matrices[i] @ bone.inverse_bind

def lerp_angle(a, b, t) -> float:
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t

class Mannequin:
    def __init__(self, player, mesh, material, foot_offset, scale, skeleton: Skeleton | None = Skeleton([])):
        self.player = player
        self.mesh = mesh
        self.material = material
        self.skeleton = skeleton
        if self.skeleton is not None and hasattr(self.skeleton, "set_rest_pose"):
             self.skeleton.set_rest_pose()
        self.foot_offset = foot_offset
        self.scale_factor = scale
        
        # visual-only yaw (radians), smoothed follow of player direction
        self.visual_yaw = 0.0
        
        # local yaw offset to give a bit of angle to the mannequin
        self.local_yaw_offset = math.radians(5.0)  # 10–25° sinnvoll

        # how fast the mannequin follows the player yaw (0.1 = soft, 0.3 = snappy)
        self.yaw_follow_strength = 0.25
        
        # Animator (optional)
        self.animator: Optional[Animator] = None

    def matrix(self):
        pos = self.player.position.copy().astype(np.float32)

        # capsule center → ground
        pos[1] -= self.player.height * self.player.stand_height

        # ground → feet
        pos[1] += self.foot_offset * self.scale_factor

        # small epsilon to avoid z-fighting / sinking
        pos[1] += 0.05

        # yaw MUST be radians
        front = self.player.front
        target_yaw = math.atan2(front[0], front[2])

        # smooth visual rotation towards player direction
        self.visual_yaw = lerp_angle(
            self.visual_yaw,
            target_yaw,
            self.yaw_follow_strength
        )

        yaw = self.visual_yaw
        yaw += self.local_yaw_offset
        c, s = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [ c, 0,  s, 0],
            [ 0, 1,  0, 0],
            [-s, 0,  c, 0],
            [ 0, 0,  0, 1],
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos

        S = np.eye(4, dtype=np.float32)
        S[0, 0] = S[1, 1] = S[2, 2] = self.scale_factor

        return T @ R @ S

    @property
    def bone_matrices(self):
        """
        Expose bone matrices for rendering.
        Returns None if this mannequin is not skinned.
        """
        if self.skeleton is None:
            return None
        return self.skeleton.bone_matrices