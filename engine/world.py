# world.py
import json
from pathlib import Path

# Note: World class is responsible for loading levels and creating game objects based on JSON data.
from gameobjects.mesh import MeshRegistry
from gameobjects.object import GameObject
from gameobjects.transform import Transform
from gameobjects.collider.aabb import AABBCollider
from gameobjects.material_lookup import Material
from gameobjects.material_lookup import MaterialRegistry

from components.audio_component import AudioComponent
from components.light_component import LightComponent
from components.physics_component import ClothComponent


class World:
    def __init__(self, audio_engine, level_path: str | None = None):
        """
        World class that holds all game objects, including static objects, lights, and the sun.

        :param self: The object itself
        """
        self.audio_engine = audio_engine
        self.objects: list[GameObject] = []
        self.static_objects: list[GameObject] = []
        self.lights: list[GameObject] = []
        self.sun: GameObject | None = None
        if level_path:
            self.load_level(level_path)

    def _create_object(self, data: dict):

        # ---------- transform ----------
        transform = Transform(
            position=data.get("position", [0, 0, 0]),
            scale=data.get("scale", [1, 1, 1])
        )

        # ---------- mesh ----------
        mesh_name = data.get("mesh")
        
        if mesh_name == "cloth":
            mesh = None  
        else:
            mesh = MeshRegistry.get(mesh_name) if mesh_name else None

        # ---------- material ----------
        material = None
        material_data = data.get("material")

        if isinstance(material_data, str):
            material = MaterialRegistry.get(material_data)

        elif isinstance(material_data, dict):
            base_name = material_data.get("name")
            if base_name:
                base_material = MaterialRegistry.get(base_name)
                material = Material(
                    double_sided=material_data.get("double_sided", False),
                    color=base_material.color,
                    texture=base_material.texture,
                    texture_scale_mode=material_data.get("texture_scale_mode", "default"),
                    texture_scale_value=material_data.get("texture_scale_value"),
                    shininess=material_data.get("shininess"),
                    specular_strength=material_data.get("specular_strength"),
                )

        # ---------- collider ----------
        collider_size = data.get("collider")
        collider = AABBCollider(size=collider_size) if collider_size else None

        # ---------- name -----------
        obj_name = data.get("obj_name")

        # ---------- object ----------
        obj = GameObject(
            transform=transform,
            mesh=mesh,
            material=material,
            collider=collider,
            obj_name=obj_name
        )

        self.objects.append(obj)

        # ---------- components ---------- 
        audio_data = data.get("audio")
        if audio_data:
            obj.add_component(AudioComponent(self.audio_engine, audio_data))

        light_data = data.get("light")
        if light_data:
            obj.add_component(LightComponent(light_data))

        cloth_data = data.get("cloth")
        if cloth_data:
            obj.add_component(
                ClothComponent(
                    width=cloth_data.get("width"),
                    height=cloth_data.get("height"),
                    segments_x=cloth_data.get("segments_x"),
                    segments_y=cloth_data.get("segments_y"),
                    gravity=cloth_data.get("gravity"),
                    wind_rate=cloth_data.get("wind_rate"),
                )
            )

        if collider is not None:
            self.static_objects.append(obj)

    def load_level(self, level_path: str):
        """
        Loads a level from a JSON file and creates game objects.

        :param self: The object itself
        :param level_path: Path to the level file
        :type level_path: str
        """
        path = Path(level_path)
        if not path.exists():
            raise FileNotFoundError(f"Level file not found: {level_path}")

        with open(path, "r") as f:
            data = json.load(f)

        objects = data.get("objects", [])
        for entry in objects:
            self._create_object(entry)
