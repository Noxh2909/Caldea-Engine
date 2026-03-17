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
    "marble": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "marble_floor.jpg"))
    ),
    "yasu": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "yasu.jpeg"))
    ),
    "sound": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "sound.png"))
    ),
    "jukebox": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "jukebox.png"))
    ),
    "laminate": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "laminate.png"))
    ),
    "wallpaper": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "wallpaper.png"))
    ),
    "stone": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "stone.png"))
    ),
    "monalisa": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "monalisa.png"))
    ),
    "wavewall": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "wavewall.png"))
    ),
    "curtain": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "curtain.png"))
    ),
    "whiteplain": lambda: Material(
        texture=Texture.load_texture(os.path.join(texture_dir, "whiteplain.png"))
    ),
    "sun": lambda: Material(color=(255, 255, 255)),
    "white": lambda: Material(color=(1.0, 1.0, 1.0)),
}


class Material:
    def __init__(
        self,
        opacity=1.0,
        double_sided=False,
        color=(1.0, 1.0, 1.0),
        texture=None,
        texture_scale_mode=None,
        texture_scale_value=None,
        shininess=32,
        specular_strength=0.5,
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
        self.opacity = opacity
        self.double_sided = double_sided
        self.color = color
        self.texture = texture
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
