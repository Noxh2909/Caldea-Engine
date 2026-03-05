from OpenGL import GL
import ctypes
import pygame
import numpy as np

from rendering.utils.renderer_utils import RenderUtils

_utils = RenderUtils()

# ============================================================
# Shader Loading
# ============================================================

# fmt: off
# debug shader sources
DEBUG_VERTEX_SHADER_SRC = _utils.load_shader("engine/rendering/debug_shader/debug.vert")
DEBUG_FRAGMENT_SHADER_SRC = _utils.load_shader("engine/rendering/debug_shader/debug.frag")

# grid plane shader sources
PLANE_VERTEX_SHADER_SRC = _utils.load_shader("engine/rendering/debug_shader/grid_plane.vert")
PLANE_FRAGMENT_SHADER_SRC = _utils.load_shader("engine/rendering/debug_shader/grid_plane.frag")
# fmt: on

# ============================================================
# Debug Renderer
# ============================================================


class DebugRenderer:
    """
    DebugRenderer
    =================

    Responsible for:
    - Rendering the 2D debug HUD overlay
    - Rendering the 3D world debug grid

    This renderer operates in two modes:
    1. World-space debug rendering (grid)
    2. Screen-space overlay rendering (HUD)
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, plane_size: float = 1000.0):
        """
        Initialize debug renderer.

        :param plane_size: Size of the debug grid plane
        """
        self.debug_enabled = True

        self._compile_shaders()
        self._cache_uniform_locations()
        self._init_debug_hud((1400, 800))

        self.grid_vao, self.grid_vertex_count = self._create_grid_plane(plane_size)
        self.model = np.identity(4, dtype=np.float32)

    # ============================================================
    # Shader Setup
    # ============================================================

    def _compile_shaders(self) -> None:
        """Compile debug and grid shader programs."""
        self.debug_program = _utils.link_program(
            DEBUG_VERTEX_SHADER_SRC, DEBUG_FRAGMENT_SHADER_SRC
        )

        self.grid_program = _utils.link_program(
            PLANE_VERTEX_SHADER_SRC, PLANE_FRAGMENT_SHADER_SRC
        )

    def _cache_uniform_locations(self) -> None:
        """Cache frequently used uniform locations."""
        self.grid_u_view = GL.glGetUniformLocation(self.grid_program, "u_view")
        self.grid_u_proj = GL.glGetUniformLocation(self.grid_program, "u_proj")
        self.grid_u_model = GL.glGetUniformLocation(self.grid_program, "u_model")

    # ============================================================
    # HUD Initialization
    # ============================================================

    def _init_debug_hud(self, viewport_size: tuple[int, int]) -> None:
        """
        Initialize screen-space HUD resources.

        :param viewport_size: (width, height) of the viewport
        """
        pygame.font.init()
        self.debug_font = pygame.font.SysFont("consolas", 16)
        self.debug_vw, self.debug_vh = viewport_size

        GL.glUseProgram(self.debug_program)

        self.debug_u_offset = GL.glGetUniformLocation(self.debug_program, "u_offset")
        self.debug_u_scale = GL.glGetUniformLocation(self.debug_program, "u_scale")
        self.debug_u_view = GL.glGetUniformLocation(self.debug_program, "u_view")

        # fmt: off
        quad = np.array([
            0, 0, 0, 1,
            1, 0, 1, 1,
            1, 1, 1, 0,
            0, 0, 0, 1,
            1, 1, 1, 0,
            0, 1, 0, 0,
        ], dtype=np.float32)
        # fmt: on

        self.debug_vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.debug_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))

        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(8))

        GL.glBindVertexArray(0)
        self.debug_tex = GL.glGenTextures(1)

    # ============================================================
    # HUD Rendering
    # ============================================================

    def render_debug_hud(
        self, clock, player, obj, obj_pos, obj_scale, extra_lines=None
    ) -> None:
        """
        Render 2D debug overlay.

        :param clock: pygame clock instance
        :param player: player object
        :param obj: controlled object dictionary
        :param obj_pos: position vector
        :param obj_scale: scale vector
        :param extra_lines: optional additional debug lines
        """
        if not self.debug_enabled:
            return

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.debug_program)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glUniform2f(self.debug_u_view, float(self.debug_vw), float(self.debug_vh))

        GL.glBindVertexArray(self.debug_vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.debug_tex)

        lines = [
            f"FPS: {clock.get_fps():.1f}",
            f"Player Pos: {player.position[0]:.2f}, {player.position[1]:.2f}, {player.position[2]:.2f}",
            f"Controlled object: {obj['target']}",
            f"Object Pos: {obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f}",
            f"Object Scale: {obj_scale[0]:.2f}, {obj_scale[1]:.2f}, {obj_scale[2]:.2f}",
        ]

        if extra_lines:
            lines.append("-----")
            lines.extend(extra_lines)

        x, y = 15, 15
        for line in lines:
            surf = self.debug_font.render(line, True, (255, 255, 255)).convert_alpha()
            w, h = self._upload_surface(surf)

            GL.glUniform2f(self.debug_u_offset, x, y)
            GL.glUniform2f(self.debug_u_scale, w, h)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            y += h + 4

        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glUseProgram(0)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    # ============================================================
    # Texture Upload
    # ============================================================

    def _upload_surface(self, surf: pygame.Surface) -> tuple[int, int]:
        """Upload pygame surface to OpenGL texture."""
        data = pygame.image.tostring(surf, "RGBA", True)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.debug_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            surf.get_width(),
            surf.get_height(),
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            data,
        )

        return surf.get_width(), surf.get_height()

    # ============================================================
    # 3D Debug Grid
    # ============================================================

    def draw_debug_grid(self, camera, aspect: float, size: float) -> None:
        """
        Render 3D debug grid plane.

        :param camera: camera instance
        :param aspect: viewport aspect ratio
        :param size: grid size
        """
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.grid_program)

        GL.glUniformMatrix4fv(self.grid_u_view, 1, GL.GL_TRUE, camera.get_view_matrix())
        GL.glUniformMatrix4fv(
            self.grid_u_proj, 1, GL.GL_TRUE, camera.get_projection_matrix(aspect)
        )
        GL.glUniformMatrix4fv(self.grid_u_model, 1, GL.GL_TRUE, self.model)

        GL.glBindVertexArray(self.grid_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.grid_vertex_count)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glBindVertexArray(0)

    # ============================================================
    # Grid Geometry
    # ============================================================

    def _create_grid_plane(self, size: float):
        """Create simple grid plane mesh."""
        # fmt: off
        vertices = np.array([
            -size, 0.0, -size,  0, 1, 0,  0, 0,
             size, 0.0, -size,  0, 1, 0,  1, 0,
             size, 0.0,  size,  0, 1, 0,  1, 1,

            -size, 0.0, -size,  0, 1, 0,  0, 0,
             size, 0.0,  size,  0, 1, 0,  1, 1,
            -size, 0.0,  size,  0, 1, 0,  0, 1,
        ], dtype=np.float32)
        # fmt: on

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )

        stride = 8 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0)
        )

        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12)
        )

        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(
            2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(24)
        )

        GL.glBindVertexArray(0)

        return vao, len(vertices) // 8
