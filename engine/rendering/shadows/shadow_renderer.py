"""Renderer for the shadow cubemap pass."""

import numpy as np
from OpenGL import GL

from gameobjects.object import GameObject
from gameobjects.player.player import look_at
from rendering.shadows.shadow_utils import ShadowPassUniforms, ShadowShaderConfig


class ShadowRenderer:
    """Render scene geometry into a point-light shadow cubemap."""

    def __init__(self, camera):
        """Create shadow rendering resources.

        Args:
            camera: Active camera used to build the cubemap projection matrices.
        """
        self.camera = camera
        self.config = ShadowShaderConfig()
        self.shadow_size = self.config.shadow_cfg.get("resolution", 1024)
        self.depth_program = self.config.depth_program
        self.uniforms = ShadowPassUniforms.from_program(self.depth_program)
        self.depth_fbo, self.depth_texture = self.create_point_shadow_map(self.shadow_size)

    def create_point_shadow_map(self, size: int) -> tuple[int, int]:
        """Allocate the framebuffer and cubemap texture for point shadows.

        Args:
            size: Width and height of each cubemap face in pixels.

        Returns:
            A tuple of ``(framebuffer_id, cubemap_texture_id)``.

        Raises:
            RuntimeError: If the framebuffer is not complete.
        """
        depth_fbo = GL.glGenFramebuffers(1)

        depth_cubemap = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, depth_cubemap)

        for i in range(6):
            GL.glTexImage2D(int(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i, 0, GL.GL_DEPTH_COMPONENT, size, size, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)

        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, depth_fbo)
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, depth_cubemap, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Shadow framebuffer not complete")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return depth_fbo, depth_cubemap

    def point_light_matrices(self, light_pos, near_plane=None, far_plane=None) -> list[np.ndarray]:
        """Build the six view-projection matrices for a point-light cubemap.

        Args:
            light_pos: World-space light position.
            near_plane: Optional near clip override.
            far_plane: Optional far clip override.

        Returns:
            A list of six matrices, one for each cubemap face.
        """
        if near_plane is None:
            near_plane = self.config.shadow_cfg.get("near_plane")
        if far_plane is None:
            far_plane = self.config.shadow_cfg.get("far_plane")

        proj = self.camera.get_projection_matrix(aspect=1.0, near=near_plane, far=far_plane, fov=90.0)

        return [
            proj @ look_at(light_pos, light_pos + np.array([1, 0, 0]), np.array([0, -1, 0])),
            proj @ look_at(light_pos, light_pos + np.array([-1, 0, 0]), np.array([0, -1, 0])),
            proj @ look_at(light_pos, light_pos + np.array([0, 1, 0]), np.array([0, 0, 1])),
            proj @ look_at(light_pos, light_pos + np.array([0, -1, 0]), np.array([0, 0, -1])),
            proj @ look_at(light_pos, light_pos + np.array([0, 0, 1]), np.array([0, -1, 0])),
            proj @ look_at(light_pos, light_pos + np.array([0, 0, -1]), np.array([0, -1, 0])),
        ]

    def render_shadow_pass(self, light_pos, scene_objects: list[GameObject], avatars=None) -> None:
        """Render scene geometry into the shadow cubemap.

        Args:
            light_pos: World-space point-light position.
            scene_objects: Static and dynamic world objects to render.
            avatars: Optional additional animated or manually transformed objects.
        """
        GL.glViewport(0, 0, self.shadow_size, self.shadow_size)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_fbo)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_FRONT)
        GL.glUseProgram(self.depth_program)

        for loc, mat in zip(self.uniforms.shadow_matrices, self.point_light_matrices(light_pos)):
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat)

        GL.glUniform1f(self.uniforms.far_plane, self.config.shadow_cfg.get("far_plane"))
        GL.glUniform3fv(self.uniforms.light_pos, 1, light_pos)

        if avatars:
            for avatar in avatars:
                mesh = getattr(avatar, "mesh", None)
                if mesh is None:
                    continue

                GL.glUniformMatrix4fv(self.uniforms.model, 1, GL.GL_TRUE, avatar.matrix())
                mesh.draw()

        for obj in scene_objects:
            if obj.mesh is None or obj.material is None:
                continue

            GL.glUniformMatrix4fv(self.uniforms.model, 1, GL.GL_TRUE, obj.transform.matrix())

            if getattr(obj.material, "double_sided", False):
                GL.glDisable(GL.GL_CULL_FACE)
            else:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_FRONT)

            obj.mesh.draw()

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
