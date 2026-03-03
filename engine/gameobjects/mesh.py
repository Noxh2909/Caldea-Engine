import numpy as np
import ctypes
from OpenGL import GL
from gameobjects.vertec import cube_vertices, sphere_vertices, cylinder_vertices


class Mesh:
    def __init__(
        self,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        *,
        positions: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        bone_ids: np.ndarray | None = None,
        bone_weights: np.ndarray | None = None,
    ):
        """
        vertices: interleaved array [pos(3), normal(3), uv(2)]
        indices: optional index buffer (uint32)
        """
        # --- Build interleaved vertex buffer ---
        if vertices is None:
            assert positions is not None and normals is not None and uvs is not None, \
                "positions, normals and uvs are required"

            assert positions.shape[0] == normals.shape[0] == uvs.shape[0], \
                "positions/normals/uvs vertex count mismatch"

            vcount = positions.shape[0]

            # Decide layout: static (8 floats) or skinned (8 floats + 4 u16 + 4 floats)
            is_skinned = bone_ids is not None and bone_weights is not None

            if is_skinned:
                assert bone_ids is not None and bone_weights is not None
                assert bone_ids.shape == (vcount, 4)
                assert bone_weights.shape == (vcount, 4)

                # positions(3) normals(3) uvs(2) weights(4)
                vertices = np.zeros((vcount, 12), dtype=np.float32)
                vertices[:, 0:3]  = positions
                vertices[:, 3:6]  = normals
                vertices[:, 6:8]  = uvs
                vertices[:, 8:12] = bone_weights

                # bone IDs stored separately (uint16)
                self.bone_ids = bone_ids.astype(np.uint16)
                self.bone_weights = bone_weights.astype(np.float32)
            else:
                vertices = np.zeros((vcount, 8), dtype=np.float32)
                vertices[:, 0:3] = positions
                vertices[:, 3:6] = normals
                vertices[:, 6:8] = uvs

                self.bone_ids = None
                self.bone_weights = None

            vertices = vertices.reshape(-1)
            self.is_skinned = is_skinned
        else:
            self.bone_ids = None
            self.bone_weights = None
            self.is_skinned = False

        stride_floats = 12 if self.is_skinned else 8
        self.vertex_count = vertices.size // stride_floats
        # Store layout info for dynamic position updates
        self.stride_floats = stride_floats
        self.stride_bytes = stride_floats * 4
        self.index_count = len(indices) if indices is not None else 0
        self.has_indices = indices is not None

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ebo = GL.glGenBuffers(1) if self.has_indices else None

        GL.glBindVertexArray(self.vao)
        # VBO
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_DYNAMIC_DRAW)

        # Keep CPU-side copy for efficient dynamic updates
        self._vertex_buffer = vertices.copy()

        # EBO (optional)
        if self.has_indices and indices is not None:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                indices.nbytes,
                indices.astype(np.uint32),
                GL.GL_STATIC_DRAW,
            )

        stride = (12 if self.is_skinned else 8) * 4

        # position (location = 0)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))

        # normal (location = 1)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))

        # uv (location = 2)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(6 * 4))

        if self.is_skinned and self.bone_ids is not None:
            # bone IDs (location = 3) -- integer attribute
            self.bone_vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.bone_vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                self.bone_ids.nbytes,
                self.bone_ids,
                GL.GL_STATIC_DRAW,
            )

            GL.glEnableVertexAttribArray(3)
            GL.glVertexAttribIPointer(
                3, 4, GL.GL_UNSIGNED_SHORT, 0, ctypes.c_void_p(0)
            )

            # bone weights (location = 4)
            GL.glEnableVertexAttribArray(4)
            GL.glVertexAttribPointer(
                4, 4, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(8 * 4)
            )

        GL.glBindVertexArray(0)

    def draw(self):
        GL.glBindVertexArray(self.vao)
        if self.has_indices:
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)


    def update_positions(self, positions: np.ndarray):
        """
        Efficient bulk update of vertex positions.
        Updates CPU copy and performs a single glBufferSubData call.
        """
        assert positions.shape[0] == self.vertex_count, \
            "Position count mismatch"

        # Update CPU-side interleaved buffer (only first 3 floats per vertex)
        reshaped = self._vertex_buffer.reshape(self.vertex_count, self.stride_floats)
        reshaped[:, 0:3] = positions

        # Single GPU upload
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER,
            0,
            self._vertex_buffer.nbytes,
            self._vertex_buffer
        )
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)


class MeshRegistry:
    """
    Docstring für MeshRegistry
    """

    _meshes: dict[str, Mesh] = {}

    @classmethod
    def get(cls, name: str) -> Mesh:
        """
        Docstring für get

        :param cls: The class itself
        :param name: The name of the mesh to retrieve
        :type name: str
        :return: The mesh object
        :rtype: Mesh
        """
        if name not in cls._meshes:
            cls._meshes[name] = cls._load_mesh(name)
        return cls._meshes[name]

    @staticmethod
    def _load_mesh(name: str) -> Mesh:
        if name == "cube":
            return Mesh(cube_vertices)
        elif name == "sphere":
            return Mesh(sphere_vertices)
        elif name == "cylinder":
            return Mesh(cylinder_vertices)
        else:
            raise ValueError(f"Unknown mesh asset: {name}")
