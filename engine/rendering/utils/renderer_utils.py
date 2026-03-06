from OpenGL import GL
import numpy as np
import math
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

    def link_program(
        self, vertex_src: str, fragment_src: str, geometry_src: Optional[str] = None
    ) -> int:
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