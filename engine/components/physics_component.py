from physics.cloth import Cloth
from gameobjects.mesh import Mesh
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gameobjects.object import GameObject


class ClothComponent:
    """
    Component wrapper that integrates Cloth simulation
    into the GameObject system.
    """

    def __init__(self, width, height, segments_x, segments_y, gravity, wind_rate):
        self.width = width
        self.height = height
        self.segments_x = segments_x
        self.segments_y = segments_y
        self.gravity = gravity
        self.wind_rate = wind_rate

        self.cloth = None
        self.game_object: Optional["GameObject"] = None

    def start(self):
        if self.game_object is None:
            raise RuntimeError(
                "ClothComponent must be attached to a GameObject before start()."
            )
        if self.game_object.transform is None:
            raise RuntimeError("GameObject must have a transform component.")

        origin = self.game_object.transform.position.copy()

        self.cloth = Cloth(
            origin=origin,
            width=self.width,
            height=self.height,
            segments_x=self.segments_x,
            segments_y=self.segments_y,
            gravity=self.gravity,
            wind_strength=self.wind_rate,
        )

        vertices, normals, uvs, indices = self.cloth.build_mesh_data()

        self.game_object.mesh = Mesh(
            positions=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
        )

    def update(self, dt):
        if self.game_object is None:
            return
        if self.cloth is None:
            return
        if self.game_object.mesh is None:
            return

        self.cloth.step(dt)

        updated_vertices = np.array(self.cloth.points, dtype=np.float32)
        self.game_object.mesh.update_positions(updated_vertices)
