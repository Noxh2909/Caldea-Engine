import numpy as np
import math


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def quat_slerp(q1, q2, t):
    dot = np.dot(q1, q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        return lerp(q1, q2, t)

    theta_0 = math.acos(dot)
    theta = theta_0 * t

    q3 = q2 - q1 * dot
    q3 /= np.linalg.norm(q3)

    return q1 * math.cos(theta) + q3 * math.sin(theta)


def compose_matrix(t, q):
    x, y, z, w = q

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,     0],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,     0],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y, 0],
        [0,                 0,                 0,                 1],
    ], dtype=np.float32)

    M = np.identity(4, dtype=np.float32)
    M[:3, 3] = t
    return M @ R


class Animator:
    def __init__(self, skeleton, clip, loop: bool = True):
        self.skeleton = skeleton
        self.clip = clip
        self.loop = loop

        self.time = 0.0
        self.duration = clip.duration

        # Debug (nur einmal ausgeben)
        print("[Animator] Init")
        print("  Bones:", len(self.skeleton.bones))
        print("  Tracks:", len(self.clip.tracks))
        print("  Track names:")
        for k in self.clip.tracks.keys():
            print("   -", k)

    def update(self, dt: float):
        if not self.clip:
            return

        # -------------------------
        # Advance time
        # -------------------------
        if self.loop:
            self.time = (self.time + dt) % self.duration
        else:
            self.time = min(self.time + dt, self.duration)

        # Debug: Zeit läuft?
        # (alle ~0.5s)
        if int(self.time * 2) % 2 == 0:
            print(f"[Animator] t = {self.time:.3f}")

        animated_bones = 0

        # -------------------------
        # Sample bones
        # -------------------------
        for i, bone in enumerate(self.skeleton.bones):

            # 🔑 WICHTIG: Track-Namen matchen (Suffix-Match)
            kfs = None
            for track_name, frames in self.clip.tracks.items():
                if track_name.endswith(bone.name):
                    kfs = frames
                    break

            if not kfs or len(kfs) < 2:
                continue

            # Keyframe interpolation
            for a, b in zip(kfs, kfs[1:]):
                if a.time <= self.time <= b.time:
                    t = (self.time - a.time) / (b.time - a.time)

                    T = lerp(a.translation, b.translation, t)
                    R = quat_slerp(a.rotation, b.rotation, t)

                    # Root translation unterdrücken
                    if bone.parent == -1:
                        T = np.zeros(3, dtype=np.float32)

                    self.skeleton.local_matrices[i] = compose_matrix(T, R)
                    animated_bones += 1
                    break

        # -------------------------
        # Apply pose
        # -------------------------
        self.skeleton.update_from_local()

        # Debug: bewegt sich überhaupt was?
        if animated_bones == 0:
            print("[Animator][WARN] No bones animated this frame!")
        else:
            print(f"[Animator] Animated bones: {animated_bones}")