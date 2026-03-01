from gameobjects.texture import Texture
import os

ENGINE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../pygame/engine
PROJECT_ROOT = os.path.dirname(ENGINE_DIR)  # .../pygame

texture_dir = os.path.join(PROJECT_ROOT, "assets", "textures")

if not os.path.isdir(texture_dir):
    raise RuntimeError(f"Texture directory not found: {texture_dir}")

MATERIAL_TABLE = {
    "wood": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "wood_wall.jpg"))
    ),
    "brick": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "brick_wall.jpg"))
    ),
    "marble": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "marble_floor.jpg"))
    ),
    "ocean": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "ocean.jpg"))
    ),
    "metal": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "metal.png"))
    ),
    "yasu": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "yasu.jpeg"))
    ),
    "js": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "js.jpeg"))
    ),
    "elian": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "elian.jpeg"))
    ),
    "sound": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "sound.png"))
    ),
    "sun": lambda: Material(color=(255, 255, 255), emissive=True),
    "white": lambda: Material(color=(1.0, 1.0, 1.0)),
}


class Material:
    def __init__(
        self,
        color=(1.0, 1.0, 1.0),
        texture=None,
        emissive=False,
        texture_scale_mode=None,
        texture_scale_value=None,
        shininess=48.0,
        specular_strength=1.0,
    ):
        """
        color         : fallback color (vec3)
        texture       : OpenGL texture id or None
        emissive      : bool
        texture_scale_mode : optional scale mode for texture coordinates
        texture_scale_value : optional scale value for texture coordinates
        shininess : optional float for shininess
        specular_strength : optional float for specular strength
        """
        self.color = color
        self.texture = texture
        self.emissive = emissive
        self.texture_scale_mode = texture_scale_mode
        self.texture_scale_value = texture_scale_value
        self.shininess = shininess
        self.specular_strength = specular_strength


class MaterialRegistry:
    _materials: dict[str, Material] = {}

    @classmethod
    def get(cls, name: str) -> Material:
        if name not in cls._materials:
            cls._materials[name] = cls._load(name)
        return cls._materials[name]

    @staticmethod
    def _load(name: str) -> Material:
        if name in MATERIAL_TABLE:
            return MATERIAL_TABLE[name]()
        else:
            raise ValueError(f"Unknown material: {name}")
