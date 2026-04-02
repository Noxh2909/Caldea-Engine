"""Helpers and configuration objects for the shadow rendering pass."""

import json
from dataclasses import dataclass

from rendering.utils.renderer_utils import RenderUtils

utils = RenderUtils()

# ============================================================
# Uniform Location Paths
# ============================================================

SHADOW_CONFIG_PATH = "engine/rendering/shadows/shadow_conifg.json"
DEPTH_VERTEX_SHADER_PATH = "engine/rendering/shadows/shader/depth.vert"
DEPTH_FRAGMENT_SHADER_PATH = "engine/rendering/shadows/shader/depth.frag"
DEPTH_GEOMETRY_SHADER_PATH = "engine/rendering/shadows/shader/depth.geom"

# ============================================================
# Shadow Rendering Utilities
# ============================================================


class ShadowShaderConfig:
    """Load and hold runtime resources for the shadow pass.

    This object owns the pass-local configuration values and the compiled depth
    shader program used by ShadowRenderer.
    """

    def __init__(self) -> None:
        """Initialize shadow config and compile the shadow program once."""
        self.load_shadow_shaders()
        self.load_shadow_config(SHADOW_CONFIG_PATH)

    def load_shadow_shaders(self) -> None:
        """Compile the depth-only shadow program for point light shadows."""
        self.depth_program = utils.link_program(utils.load_shader(DEPTH_VERTEX_SHADER_PATH), utils.load_shader(DEPTH_FRAGMENT_SHADER_PATH), utils.load_shader(DEPTH_GEOMETRY_SHADER_PATH))

    def load_shadow_config(self, path: str) -> None:
        """Load the pass-local shadow configuration from disk.

        Args:
            path: Path to the JSON file containing the ``shadow`` block.
        """
        with open(path, "r") as f:
            cfg = json.load(f)
        self.shadow_cfg = cfg.get("shadow", {})


# ============================================================
# Shadow Pass Uniforms Dataclass
# ============================================================


@dataclass(slots=True)
class ShadowPassUniforms:
    """Uniform locations used by the point-light shadow pass."""

    model: int
    light_pos: int
    far_plane: int
    shadow_matrices: list[int]

    @classmethod
    def from_program(cls, program: int) -> "ShadowPassUniforms":
        """Factory method to create ShadowPassUniforms from shader program."""
        return cls(
            model=utils.require_uniform(program, "model"),
            light_pos=utils.require_uniform(program, "lightPos"),
            far_plane=utils.require_uniform(program, "far_plane"),
            shadow_matrices=utils.require_uniform_array(program, "shadowMatrices", 6),
        )
