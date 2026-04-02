"""Helpers and configuration objects for the SSAO rendering pass."""

import json
from dataclasses import dataclass

from rendering.utils.renderer_utils import RenderUtils

utils = RenderUtils()

# ============================================================
# Uniform Location Paths
# ============================================================

SSAO_CONFIG_PATH = "engine/rendering/ssao/ssao_config.json"
SSAO_GEOMETRY_VERTEX_SHADER_PATH = "engine/rendering/ssao/shader/geometry.vert"
SSAO_GEOMETRY_FRAGMENT_SHADER_PATH = "engine/rendering/ssao/shader/geometry.frag"
SSAO_BLUR_FRAGMENT_SHADER_PATH = "engine/rendering/ssao/shader/ssao_blur.frag"
SSAO_FRAGMENT_SHADER_PATH = "engine/rendering/ssao/shader/ssao.frag"
SSAO_FULLSCREEN_VERTEX_SHADER_PATH = "engine/rendering/ssao/shader/fullscreen.vert"

# ============================================================
# SSAO Rendering Utilities
# ============================================================


class SSAOShaderConfig:
    """Load and hold runtime resources for the SSAO pass."""

    def __init__(self) -> None:
        """Initialize pass-local configuration and compile programs once."""
        self.load_ssao_shaders()
        self.load_ssao_config(SSAO_CONFIG_PATH)

    def load_ssao_shaders(self) -> None:
        """Compile all shader programs used by the SSAO pass."""
        self.geometry_program = utils.link_program(utils.load_shader(SSAO_GEOMETRY_VERTEX_SHADER_PATH), utils.load_shader(SSAO_GEOMETRY_FRAGMENT_SHADER_PATH))
        self.blur_program = utils.link_program(utils.load_shader(SSAO_FULLSCREEN_VERTEX_SHADER_PATH), utils.load_shader(SSAO_BLUR_FRAGMENT_SHADER_PATH))
        self.ssao_program = utils.link_program(utils.load_shader(SSAO_FULLSCREEN_VERTEX_SHADER_PATH), utils.load_shader(SSAO_FRAGMENT_SHADER_PATH))

    def load_ssao_config(self, path: str) -> None:
        """Load the pass-local SSAO configuration from disk."""
        with open(path, "r", encoding="utf-8") as file:
            cfg = json.load(file)
        self.ssao_cfg = cfg.get("ssao", {})


# ============================================================
# SSAO Pass Uniforms Dataclasses
# ============================================================


@dataclass(slots=True)
class GeometryPassUniforms:
    """Uniform locations used during the geometry G-buffer pass."""

    model: int
    view: int
    projection: int
    object_color: int

    @classmethod
    def from_program(cls, program: int) -> "GeometryPassUniforms":
        """Create geometry-pass uniform bindings from a linked program."""
        return cls(
            model=utils.require_uniform(program, "model"),
            view=utils.require_uniform(program, "view"),
            projection=utils.require_uniform(program, "projection"),
            object_color=utils.require_uniform(program, "objectColor"),
        )


@dataclass(slots=True)
class SSAOPassUniforms:
    """Uniform locations used during the SSAO evaluation pass."""

    g_position: int
    g_normal: int
    noise_tex: int
    projection: int
    kernel_size: int
    radius: int
    bias: int
    noise_scale: int
    samples: list[int]

    @classmethod
    def from_program(cls, program: int, sample_count: int) -> "SSAOPassUniforms":
        """Create SSAO-pass uniform bindings from a linked program."""
        return cls(
            g_position=utils.require_uniform(program, "gPosition"),
            g_normal=utils.require_uniform(program, "gNormal"),
            noise_tex=utils.require_uniform(program, "noiseTex"),
            projection=utils.require_uniform(program, "projection"),
            kernel_size=utils.require_uniform(program, "kernelSize"),
            radius=utils.require_uniform(program, "radius"),
            bias=utils.require_uniform(program, "bias"),
            noise_scale=utils.require_uniform(program, "noiseScale"),
            samples=utils.require_uniform_array(program, "samples", sample_count),
        )


@dataclass(slots=True)
class SSAOBlurUniforms:
    """Uniform locations used during the SSAO blur pass."""

    ssao_input: int
    texel_size: int

    @classmethod
    def from_program(cls, program: int) -> "SSAOBlurUniforms":
        """Create blur-pass uniform bindings from a linked program."""
        return cls(ssao_input=utils.require_uniform(program, "ssaoInput"), texel_size=utils.require_uniform(program, "texelSize"))
