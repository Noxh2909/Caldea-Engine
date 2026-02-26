import numpy as np

class Keyframe:
    def __init__(self, time, translation, rotation):
        self.time = time
        self.translation = np.array(translation, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)  # quat


class AnimationTrack:
    def __init__(self, bone_name, keyframes):
        self.bone_name = bone_name
        self.keyframes = keyframes


class AnimationClip:
    def __init__(self, name, duration, fps, tracks):
        self.name = name
        self.duration = duration
        self.fps = fps
        self.tracks = tracks  # bone_name -> AnimationTrack