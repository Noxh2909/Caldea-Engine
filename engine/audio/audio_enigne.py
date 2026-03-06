from openal import Listener, oalQuit
from openal.al import alDistanceModel, AL_INVERSE_DISTANCE_CLAMPED


class AudioEngine:
    """
    Minimal Audio Engine.
    Handles:
        - Listener position sync
        - Updating registered AudioSources
    """

    def __init__(self):
        self.listener = Listener()
        self.sources = []
        self.object_sources = {}

    # ----------------------------
    # Source Management
    # ----------------------------
    def add_source(self, source):
        self.sources.append(source)

    def remove_source(self, source):
        if source in self.sources:
            self.sources.remove(source)

    # ----------------------------
    # Listener Update
    # ----------------------------
    def update_listener(self, camera):
        # Position = echtes Eye (nicht nur Player-Position)
        eye = camera.player.position.copy()
        eye[1] += camera.player.height * camera.eye_height_factor
        eye[1] += camera.player._headbob_offset
        eye[1] += camera.fps_eye_vertical_bias

        self.listener.set_position(eye)

        # Richtige Blickrichtung aus Player
        forward = camera.player.front
        up = camera.player.up

        self.listener.set_orientation(
            (
                float(forward[0]),
                float(forward[1]),
                float(forward[2]),
                float(up[0]),
                float(up[1]),
                float(up[2]),
            )
        )

    # ----------------------------
    # Per-frame Update
    # ----------------------------
    def update(self, camera):
        # Ensure distance model is set after context is active
        if not hasattr(self, "_distance_model_set"):
            alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED)
            self._distance_model_set = True

        self.update_listener(camera)

        for source in self.sources:
            source.update()

    # ----------------------------
    # Shutdown
    # ----------------------------
    def shutdown(self):
        for source in self.sources:
            source.stop()
        oalQuit()
