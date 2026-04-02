from audio.audio_source import AudioSource
import os


class AudioComponent:
    """
    Component for handling 3D audio sources attached to GameObjects.
    """

    def __init__(self, audio_engine, config):
        self.audio_engine = audio_engine
        self.config = config
        self.source = None
        self.game_object = None

    def start(self):
        if self.game_object is None:
            raise RuntimeError("AudioComponent must be attached to GameObject")

        full_path = os.path.join("engine/audio/audiosamples", self.config.get("path"))

        self.source = AudioSource(path=full_path, position=self.game_object.transform.position)

        # --- Configuration ---
        loop = self.config.get("loop")

        # Volume in JSON is 0–100 → convert to 0–1
        volume = self.config.get("volume")
        gain = max(0.0, min(1.0, volume / 100.0))

        max_distance = self.config.get("max_distance")
        rolloff = self.config.get("rolloff")

        self.source.set_gain(gain)
        self.source.set_max_distance(max_distance)
        self.source.set_rolloff(rolloff)

        self.audio_engine.add_source(self.source)

        self.source.play(loop=loop)

    def update(self, dt):
        if self.source and self.game_object:
            self.source.set_position(self.game_object.transform.position)

    def stop(self):
        if self.source:
            self.source.stop()
