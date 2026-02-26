# world.py
import json
from pathlib import Path

from gameobjects.mesh import Mesh, MeshRegistry
from gameobjects.object import GameObject
from gameobjects.transform import Transform
from gameobjects.collider.aabb import AABBCollider
from gameobjects.material_lookup import Material
from gameobjects.material_lookup import MaterialRegistry


class World:
    def __init__(self, level_path: str | None = None):
        """
        Docstring für __init__

        :param self: The object itself
        """
        self.objects: list[GameObject] = []
        self.static_objects: list[GameObject] = []
        self.lights: list[GameObject] = []
        self.sun: GameObject | None = None
        if level_path:
            self.load_level(level_path)

    def _create_object(self, data: dict):
        """
        Docstring für _create_object

        :param self: The object itself
        :param data: dict with object parameters
        :type data: dict
        """
        # ---------- transform ----------
        position = data.get("position", [0, 0, 0])
        scale = data.get("scale", [1, 1, 1])
        transform = Transform(position=position, scale=scale)
        transform.position[1] += transform.scale[1] * 0.5 

        # ---------- mesh ----------
        mesh = None
        mesh_name = data.get("mesh")
        if mesh_name:
            mesh = MeshRegistry.get(mesh_name)

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
                    color=base_material.color,
                    texture=base_material.texture,
                    emissive=base_material.emissive,
                    texture_scale_mode=material_data.get(
                        "texture_scale_mode", "default"
                    ),
                    texture_scale_value=material_data.get("texture_scale_value"),
                )

        # ---------- collider ----------
        collider = None
        collider_size = data.get("collider")
        if collider_size:
            collider = AABBCollider(size=collider_size)

        # ---------- light ----------
        light = data.get("light")

        obj = GameObject(
            mesh=mesh,
            transform=transform,
            material=material,
            collider=collider,
            light=light,
        )

        self.objects.append(obj)

        # Light-emitting objects (can be multiple)
        if light is not None:
            self.lights.append(obj)
            if self.sun is None:
                self.sun = obj

        if collider is not None:
            self.static_objects.append(obj)

    def load_level(self, level_path: str):
        """
        Docstring für load_level

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
