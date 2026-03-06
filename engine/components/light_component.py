from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gameobjects.object import GameObject


class LightComponent:
    def __init__(self, config: dict):
        self.config = config
        self.game_object: Optional["GameObject"] = None

    def get_light_data(self):
        if self.game_object is None:
            return None

        return {
            "position": self.game_object.transform.position,
            "direction": self.config.get("direction"),
            "color": self.config.get("color"),
            "intensity": self.config.get("intensity"),
            "ambient": self.config.get("ambient_strength"),
        }
