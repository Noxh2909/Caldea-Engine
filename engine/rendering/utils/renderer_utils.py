from OpenGL import GL
import json
from typing import Optional

# ============================================================
# Render Utilities
# ============================================================


class RenderUtils:
    """
    RenderUtils
    =================

    Central utility helper for the rendering pipeline.

    Responsibilities:
    - Shader loading and compilation
    - Program linking
    - Renderer configuration loading (JSON)
    - Projection matrix utilities

    This class contains no rendering state —
    only reusable helper functionality.
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self) -> None:
        """
        Initialize utility system and load renderer configuration.

        Loads renderer_config.json and exposes configuration
        dictionaries for bloom, SSAO, shadow, and volumetric passes.
        """
        self.load_renderer_config("engine/rendering/renderer_config.json")

    # ============================================================
    # Shader Utilities
    # ============================================================

    def load_shader(self, path: str) -> str:
        """
        Load shader source from file.

        :param path: File path to shader source
        :return: Shader source as string
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ============================================================
    # Configuration Loading
    # ============================================================

    def load_renderer_config(self, path: str) -> None:
        """
        Load renderer configuration from JSON file.

        Extracts configuration blocks for:
        - Bloom
        - SSAO
        - Shadow
        - Volumetric lighting

        :param path: Path to configuration JSON
        """
        with open(path, "r") as f:
            cfg = json.load(f)

        self.bloom_cfg = cfg.get("bloom", {})
        self.ssao_cfg = cfg.get("ssao", {})
        self.shadow_cfg = cfg.get("shadow", {})
        self.volumetric_cfg = cfg.get("volumetric", {})

    # ============================================================
    # Shader Compilation
    # ============================================================

    def compile_shader(self, source: str, shader_type: int) -> int:
        """
        Compile individual OpenGL shader.

        :param source: GLSL source code
        :param shader_type: OpenGL shader type enum
        :return: Compiled shader ID
        :raises RuntimeError: If compilation fails
        """
        shader = GL.glCreateShader(shader_type)
        if shader is None or shader == 0:
            raise RuntimeError("Failed to create shader")

        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)

        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            info = GL.glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compilation failed: {info}")

        return shader

    def link_program(self, vertex_src: str, fragment_src: str, geometry_src: Optional[str] = None) -> int:
        """
        Link OpenGL shader program.

        Attaches vertex + fragment shaders and
        optionally a geometry shader.

        :param vertex_src: Vertex shader source
        :param fragment_src: Fragment shader source
        :param geometry_src: Optional geometry shader source
        :return: Linked program ID
        :raises RuntimeError: If linking fails
        """
        program = GL.glCreateProgram()
        if program is None or program == 0:
            raise RuntimeError("Failed to create program")

        vs = self.compile_shader(vertex_src, GL.GL_VERTEX_SHADER)
        fs = self.compile_shader(fragment_src, GL.GL_FRAGMENT_SHADER)

        GL.glAttachShader(program, vs)
        GL.glAttachShader(program, fs)

        gs = None
        if geometry_src:
            gs = self.compile_shader(geometry_src, GL.GL_GEOMETRY_SHADER)
            GL.glAttachShader(program, gs)

        GL.glLinkProgram(program)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            info = GL.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Program linking failed: {info}")

        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        if gs:
            GL.glDeleteShader(gs)

        return program

    def require_uniform(self, program: int, name: str) -> int:
        """Return a required uniform location for an OpenGL program.

        Args:
            program: Linked OpenGL program id.
            name: Uniform name inside the shader.

        Returns:
            The resolved uniform location.

        Raises:
            RuntimeError: If the uniform cannot be resolved.
        """
        location = GL.glGetUniformLocation(program, name)
        if location == -1:
            raise RuntimeError(f"Uniform '{name}' nicht im Shader gefunden")
        return location

    def require_uniform_array(self, program: int, base_name: str, count: int) -> list[int]:
        """Return uniform locations for a shader uniform array.

        Args:
            program: Linked OpenGL program id.
            base_name: Base array name, for example ``shadowMatrices``.
            count: Number of array elements to resolve.

        Returns:
            A list of uniform locations in array order.
        """
        locations = []
        for i in range(count):
            name = f"{base_name}[{i}]"
            locations.append(self.require_uniform(program, name))
        return locations
