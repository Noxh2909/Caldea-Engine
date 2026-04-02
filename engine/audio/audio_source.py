from openal import oalOpen
import numpy as np


class AudioSource:
    """
    Minimal 3D audio source.
    Handles:
        - Playback
        - 3D position
        - Distance-based attenuation (handled by OpenAL)
    """

    def __init__(self, path, position=(0.0, 0.0, 0.0)):
        self.source = oalOpen(path)
        self.position = list(position)

        # Spatial configuration
        if self.source is not None:
            self.source.set_position(self.position)
            self.source.set_rolloff_factor(1.0)
            self.source.set_reference_distance(1.0)
            self.source.set_max_distance(10.0)

    # ----------------------------
    # Playback
    # ----------------------------
    def play(self, loop=True):
        if self.source is not None:
            self.source.set_looping(loop)
            self.source.play()

    def stop(self):
        if self.source is not None:
            self.source.stop()

    # ----------------------------
    # Spatial Update
    # ----------------------------
    def set_position(self, position):
        self.position = list(position)
        if self.source is not None:
            self.source.set_position(self.position)

    def update(self):
        if self.source is not None:
            self.source.set_position(self.position)

    # ----------------------------
    # Soft distance fade-out
    # ----------------------------
    def apply_distance_fade(self, listener_pos, max_distance, base_gain=1.0, fade_start_ratio=0.4):
        """
        Smoothly fades out sound between fade_start and max_distance.
        fade_start_ratio defines where fading begins (0.0–1.0 of max_distance).
        """
        if self.source is None:
            return

        distance = np.linalg.norm(np.array(listener_pos) - np.array(self.position))

        # Clamp distance to [0, max_distance]
        d = max(0.0, min(max_distance, distance))

        # Normalize distance (0 at source, 1 at max_distance)
        t = d / max_distance

        # Invert so 0 distance = 1 volume, max_distance = 0 volume
        t = 1.0 - t

        # Smoothstep curve for natural fade (both fade-in and fade-out)
        smooth = t * t * (3.0 - 2.0 * t)

        self.source.set_gain(base_gain * smooth)

    # ----------------------------
    # Configuration Setters
    # ----------------------------
    def set_loop(self, loop):
        if self.source is not None:
            self.source.set_looping(loop)

    def set_gain(self, gain):
        if self.source is not None:
            self.source.set_gain(gain)

    def set_max_distance(self, max_distance):
        if self.source is not None:
            self.source.set_max_distance(max_distance)

    def set_rolloff(self, rolloff):

        if self.source is not None:
            self.source.set_rolloff_factor(rolloff)
