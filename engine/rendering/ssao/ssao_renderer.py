"""Renderer for the SSAO geometry, sampling and blur passes."""

import random

import numpy as np
from OpenGL import GL

from gameobjects.object import GameObject
from rendering.ssao.ssao_utils import GeometryPassUniforms, SSAOBlurUniforms, SSAOShaderConfig, SSAOPassUniforms


class SSAORenderer:
    """Render screen-space ambient occlusion and expose the blurred result."""

    def __init__(self, camera, width: int, height: int):
        """Create all SSAO resources.

        Args:
            camera: Active camera used for projection and view matrices.
            width: Target framebuffer width.
            height: Target framebuffer height.
        """
        self.camera = camera
        self.width = width
        self.height = height
        self.config = SSAOShaderConfig()
        self.sample_count = int(self.config.ssao_cfg.get("kernel_size", 64))

        self.geometry_program = self.config.geometry_program
        self.ssao_program = self.config.ssao_program
        self.blur_program = self.config.blur_program

        self.geometry_uniforms = GeometryPassUniforms.from_program(self.geometry_program)
        self.ssao_uniforms = SSAOPassUniforms.from_program(self.ssao_program, self.sample_count)
        self.blur_uniforms = SSAOBlurUniforms.from_program(self.blur_program)

        self.create_ssao_buffers(width, height)
        self.ssao_kernel = self.create_ssao_kernel(self.sample_count)
        self.ssao_noise = self.generate_ssao_noise()
        self.ssao_noise_texture = self.create_noise_texture(self.ssao_noise)

        self.upload_static_uniforms()

    @property
    def blur_texture(self) -> int:
        """Return the final blurred SSAO texture for the composition pass."""
        return self.ssao_blur_texture

    def create_ssao_buffers(self, width: int, height: int) -> None:
        """Allocate the G-buffer, SSAO framebuffer and blur framebuffer."""
        self.g_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.g_fbo)

        self.g_position = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_position)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, width, height, 0, GL.GL_RGB, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.g_position, 0)

        self.g_normal = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_normal)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, width, height, 0, GL.GL_RGB, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self.g_normal, 0)

        self.g_color = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_color)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, width, height, 0, GL.GL_RGB, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D, self.g_color, 0)

        self.rbo_depth = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo_depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.rbo_depth)

        attachments = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2]
        GL.glDrawBuffers(len(attachments), attachments)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("G-Buffer Framebuffer not complete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.ssao_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_fbo)
        self.ssao_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RED, width, height, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.ssao_texture, 0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("SSAO Framebuffer not complete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.ssao_blur_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_blur_fbo)
        self.ssao_blur_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_blur_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RED, width, height, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.ssao_blur_texture, 0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("SSAO Blur Framebuffer not complete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def create_ssao_kernel(self, kernel_size: int) -> list[np.ndarray]:
        """Generate hemisphere sample vectors for the SSAO shader."""
        kernel = []
        for i in range(kernel_size):
            sample = self._random_unit_vector(positive_z=True)
            scale = float(i) / kernel_size
            scale = 0.1 + 0.9 * (scale * scale)
            kernel.append(sample * scale)
        return kernel

    def generate_ssao_noise(self) -> np.ndarray:
        """Generate the tiled noise texture data used to rotate SSAO samples."""
        noise = np.zeros((4, 4, 3), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                noise[i, j] = self._random_unit_vector(positive_z=False)
        return noise

    def _random_unit_vector(self, positive_z: bool) -> np.ndarray:
        """Return a non-zero normalized random vector for SSAO sampling."""
        while True:
            z_min, z_max = (0.0, 1.0) if positive_z else (0.0, 0.0)
            vector = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(z_min, z_max)], dtype=np.float32)
            length = np.linalg.norm(vector)
            if length > 1e-6:
                return vector / length

    def create_noise_texture(self, noise: np.ndarray) -> int:
        """Create the OpenGL texture that stores the SSAO rotation noise."""
        noise_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, noise_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, 4, 4, 0, GL.GL_RGB, GL.GL_FLOAT, noise)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        return noise_texture

    def upload_static_uniforms(self) -> None:
        """Upload SSAO constants that do not change per frame."""
        GL.glUseProgram(self.ssao_program)
        for location, sample in zip(self.ssao_uniforms.samples, self.ssao_kernel):
            GL.glUniform3fv(location, 1, sample)
        GL.glUniform1i(self.ssao_uniforms.kernel_size, len(self.ssao_kernel))
        GL.glUniform1i(self.ssao_uniforms.g_position, 0)
        GL.glUniform1i(self.ssao_uniforms.g_normal, 1)
        GL.glUniform1i(self.ssao_uniforms.noise_tex, 2)

        GL.glUseProgram(self.blur_program)
        GL.glUniform1i(self.blur_uniforms.ssao_input, 0)

    def render_ssao_pass(self, camera, scene_objects: list[GameObject], quad_vao: int) -> None:
        """Render the geometry, SSAO evaluation and blur passes.

        Args:
            camera: Active camera for the current frame.
            scene_objects: Visible scene objects with geometry.
            quad_vao: Fullscreen-quad vertex array object.
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.g_fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(int(GL.GL_COLOR_BUFFER_BIT) | int(GL.GL_DEPTH_BUFFER_BIT))
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self.geometry_program)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.width / self.height)

        GL.glUniformMatrix4fv(self.geometry_uniforms.view, 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(self.geometry_uniforms.projection, 1, GL.GL_TRUE, projection)

        for obj in scene_objects:
            if obj.mesh is None or obj.material is None:
                continue

            GL.glUniformMatrix4fv(self.geometry_uniforms.model, 1, GL.GL_TRUE, obj.transform.matrix())
            GL.glUniform3f(self.geometry_uniforms.object_color, 1.0, 1.0, 1.0)

            if getattr(obj.material, "double_sided", False):
                GL.glDisable(GL.GL_CULL_FACE)
            else:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_BACK)

            obj.mesh.draw()

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self.ssao_program)
        GL.glUniformMatrix4fv(self.ssao_uniforms.projection, 1, GL.GL_TRUE, projection)
        GL.glUniform1f(self.ssao_uniforms.radius, self.config.ssao_cfg.get("radius", 0.8))
        GL.glUniform1f(self.ssao_uniforms.bias, self.config.ssao_cfg.get("bias", 0.01))
        GL.glUniform2f(self.ssao_uniforms.noise_scale, self.width / 4.0, self.height / 4.0)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_position)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_normal)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_noise_texture)

        GL.glBindVertexArray(quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_blur_fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.blur_program)
        GL.glUniform2f(self.blur_uniforms.texel_size, 1.0 / self.width, 1.0 / self.height)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_texture)

        GL.glBindVertexArray(quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
