import ctypes
import random
import numpy as np
from OpenGL import GL

from gameobjects.player.player import look_at
from rendering.utils.renderer_utils import RenderUtils

utils = RenderUtils()

# =========================
# Shader Sources
# =========================

# fmt: off
# Depth Shader for shadow mapping
DEPTH_VERTEX_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/depth.vert")
DEPTH_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/depth.frag")
DEPTH_GEOMETRY_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/depth.geom")

# Geometry Pass Shader for SSAO
GEOMETRY_VERTEX_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/geometry.vert")
GEOMETRY_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/geometry.frag")

# SSAO Shader
SSAO_VERTEX_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/ssao.vert")
SSAO_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/ssao.frag")
SSAO_BLUR_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/ssao_blur.frag")

# VOLUMETRIC LIGHTING Shader
VOLUMETRIC_LIGHT_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/volumetric.frag")

# BLOOM Shader
BLOOM_BLUR_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/bloom_blur.frag")
BLOOM_BRIGHT_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/bloom_bright.frag")
BLOOM_FINAL_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/bloom_final.frag")

# Final Shader for rendering to screen
FINAL_VERTEX_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/final.vert")
FINAL_FRAGMENT_SHADER_SRC = utils.load_shader("engine/rendering/lighting_shader/final.frag")
# fmt: on

# =========================
# Renderer Class
# =========================


class RenderObject:
    def __init__(self, mesh, transform, material):
        """
        Lightweight container used by the lighting pipeline.

        Encapsulates:
        - Mesh geometry
        - World transform
        - Material definition

        This object is rendering-only and contains no gameplay logic.
        """
        self.mesh = mesh
        self.transform = transform
        self.material = material


class LightRenderer:
    def __init__(self, camera, width=1400, height=800):
        """
        Initialize the lighting renderer.

        Responsible for:
        - Shadow mapping
        - SSAO
        - HDR rendering
        - Bloom
        - Volumetric lighting
        - Final composition pass

        :param width: Framebuffer width
        :param height: Framebuffer height
        """
        self.width = width
        self.height = height
        self.camera = camera
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

        # compile shaders
        self.compile_lighting_shaders()

        # create frame buffers and textures
        self.create_frame_buffers()
        self.create_point_shadow_map(1024)
        # create HDR and bloom buffers
        self.create_hdr_bloom_buffers()

        # SSAO sampling kernel, upload noise texture
        self.ssao_kernel = self.create_ssao_kernel(64)
        self.ssao_noise = self.generate_ssao_noise()
        self.ssao_noise_texture = self.create_noise_texture(self.ssao_noise)

        # cache frequently used uniform locations
        self.cache_uniform_locations()

        # set up projection matrices
        # self.projection = utils.perspective(
        #     math.radians(120.0), self.width / self.height, 0.1, 100.0
        # )

        self.create_fullscreen_quad()

        # model matrix for the grid plane
        self.model = np.identity(4, dtype=np.float32)

        # precompute SSAO sample kernel
        GL.glUseProgram(self.ssao_program)
        for i, sample in enumerate(self.ssao_kernel):
            loc = GL.glGetUniformLocation(self.ssao_program, f"samples[{i}]")
            GL.glUniform3fv(loc, 1, sample)
        GL.glUniform1i(
            GL.glGetUniformLocation(self.ssao_program, "kernelSize"),
            len(self.ssao_kernel),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.ssao_program, "radius"),
            utils.ssao_cfg.get("radius"),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.ssao_program, "bias"),
            utils.ssao_cfg.get("bias"),
        )
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.ssao_program, "projection"),
            1,
            GL.GL_TRUE,
            self.camera.get_projection_matrix(self.width / self.height),
        )

    # -----------------------
    # Shader Compilation
    # -----------------------
    # Functions to compile all lighting shaders used in the renderer.
    # -----------------------

    def compile_lighting_shaders(self):
        """
        Compiles all the lighting shaders used in the renderer.

        :param self: The object itself
        """
        self.depth_program = utils.link_program(
            DEPTH_VERTEX_SHADER_SRC,
            DEPTH_FRAGMENT_SHADER_SRC,
            DEPTH_GEOMETRY_SHADER_SRC,
        )
        self.geometry_program = utils.link_program(
            GEOMETRY_VERTEX_SHADER_SRC, GEOMETRY_FRAGMENT_SHADER_SRC
        )
        self.ssao_program = utils.link_program(
            SSAO_VERTEX_SHADER_SRC, SSAO_FRAGMENT_SHADER_SRC
        )
        self.ssao_blur_program = utils.link_program(
            SSAO_VERTEX_SHADER_SRC, SSAO_BLUR_FRAGMENT_SHADER_SRC
        )
        self.bloom_bright_program = utils.link_program(
            SSAO_VERTEX_SHADER_SRC,
            BLOOM_BRIGHT_FRAGMENT_SHADER_SRC,
        )
        self.bloom_blur_program = utils.link_program(
            SSAO_VERTEX_SHADER_SRC,
            BLOOM_BLUR_FRAGMENT_SHADER_SRC,
        )
        self.bloom_final_program = utils.link_program(
            SSAO_VERTEX_SHADER_SRC, BLOOM_FINAL_FRAGMENT_SHADER_SRC
        )
        self.volumetric_program = utils.link_program(
            SSAO_VERTEX_SHADER_SRC, VOLUMETRIC_LIGHT_FRAGMENT_SHADER_SRC
        )
        self.final_program = utils.link_program(
            FINAL_VERTEX_SHADER_SRC, FINAL_FRAGMENT_SHADER_SRC
        )

    # -----------------------
    # Shadow Map Creation
    # -----------------------
    # Functions to create shadow maps for point lights.
    # -----------------------

    def create_point_shadow_map(self, size: int):
        """
        Docstring für create_point_shadow_map

        :param size: The size (width and height) of the shadow map texture
        :type size: int
        """
        depth_fbo = GL.glGenFramebuffers(1)

        depth_cubemap = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, depth_cubemap)

        for i in range(6):
            GL.glTexImage2D(
                int(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X) + i,
                0,
                GL.GL_DEPTH_COMPONENT,
                size,
                size,
                0,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
                None,
            )

        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, depth_fbo)
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, depth_cubemap, 0
        )
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        assert (
            GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        return depth_fbo, depth_cubemap

    # -----------------------
    # Light Setup
    # -----------------------
    # Functions to set up lighting parameters.
    # -----------------------

    def set_light(self, direction, color, intensity, ambient=None, position=None):
        """
        Configure active scene light.

        :param direction: Light direction (normalized internally)
        :param color: RGB light color
        :param intensity: Direct light intensity
        :param ambient: Ambient light strength
        :param position: World-space light position (required for point shadows)
        """

        self.light_dir = np.array(direction, dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)

        self.light_color = np.array(color, dtype=np.float32)
        self.light_pos = position

        self.light_intensity = intensity
        self.light_ambient = ambient

    # -----------------------
    # HDR and Bloom Buffer Creation
    # -----------------------
    # Functions to create HDR and bloom framebuffers and textures.
    # -----------------------

    def create_hdr_bloom_buffers(self):
        """
        Create framebuffers and textures needed for HDR rendering and bloom effect.
        """
        # =========================
        # HDR FBO (Color + Depth Texture)
        # =========================
        self.hdr_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.hdr_fbo)

        # HDR color texture
        self.hdr_color = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hdr_color)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.hdr_color,
            0,
        )

        # Depth texture (IMPORTANT for volumetric lighting)
        self.hdr_depth = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hdr_depth)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT24,
            self.width,
            self.height,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self.hdr_depth,
            0,
        )

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("HDR Framebuffer not complete")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # =========================
        # Volumetric Lighting FBO
        # =========================
        self.volumetric_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.volumetric_fbo)

        self.volumetric_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.volumetric_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.volumetric_tex,
            0,
        )

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Volumetric Framebuffer not complete")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # =========================
        # Bloom Ping-Pong Buffers
        # =========================
        self.blur_fbo = GL.glGenFramebuffers(2)
        self.blur_tex = GL.glGenTextures(2)

        for i in range(2):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.blur_fbo[i])
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.blur_tex[i])
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                self.width,
                self.height,
                0,
                GL.GL_RGBA,
                GL.GL_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                self.blur_tex[i],
                0,
            )

            if (
                GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                != GL.GL_FRAMEBUFFER_COMPLETE
            ):
                raise RuntimeError("Bloom Framebuffer not complete")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # -----------------------
    # SSAO Buffer Creation
    # -----------------------
    # Functions to create SSAO framebuffers and textures.
    # -----------------------

    def create_ssao_buffers(self, width: int, height: int) -> dict:
        """Create framebuffers and textures needed for SSAO.

        The SSAO pass samples positions and normals from a G-buffer.  We store
        positions in view space, normals and a colour buffer (unused here
        but often useful).  An additional texture holds the SSAO
        occlusion factor and another stores a blurred version of that
        texture.

        :param width: Width of the screen.
        :param height: Height of the screen.
        :return: Dictionary with names -> texture ids and FBOs.
        """
        self.g_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.g_fbo)

        # Position texture
        self.g_position = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_position)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB32F,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.g_position,
            0,
        )

        # Normal texture
        self.g_normal = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_normal)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB16F,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_TEXTURE_2D,
            self.g_normal,
            0,
        )

        # Color Buffer (RGB16F for view normal)
        self.g_color = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_color)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB16F,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT2,
            GL.GL_TEXTURE_2D,
            self.g_color,
            0,
        )

        # depth renderbuffer
        self.rbo_depth = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo_depth)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, width, height
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            self.rbo_depth,
        )

        # specify the color attachments for rendering
        self.attachments = [
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_COLOR_ATTACHMENT2,
        ]
        GL.glDrawBuffers(len(self.attachments), self.attachments)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("G-Buffer Framebuffer not complete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # SSAO FBO and texture
        self.ssao_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_fbo)
        self.ssao_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RED,
            width,
            height,
            0,
            GL.GL_RED,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.ssao_texture,
            0,
        )
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("SSAO Framebuffer not complete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # SSAO Blur FBO and texture
        self.ssao_blur_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_blur_fbo)
        self.ssao_blur_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_blur_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RED,
            width,
            height,
            0,
            GL.GL_RED,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.ssao_blur_texture,
            0,
        )
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("SSAO Blur Framebuffer not complete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        return {
            "g_fbo": self.g_fbo,
            "g_position": self.g_position,
            "g_normal": self.g_normal,
            "g_color": self.g_color,
            "ssao_fbo": self.ssao_fbo,
            "ssao_tex": self.ssao_texture,
            "ssao_blur_fbo": self.ssao_blur_fbo,
            "ssao_blur_texture": self.ssao_blur_texture,
            "depth_rbo": self.rbo_depth,
        }

    # -----------------------
    # SSAO Kernel Generation
    # -----------------------
    # Functions to generate SSAO sample kernel.
    # -----------------------

    def create_ssao_kernel(self, kernel_size: int = 64) -> list:
        """Generate a list of sample vectors for SSAO.

        The samples are distributed within a hemisphere oriented along the
        z-axis.  A bias towards the origin is applied so more samples lie
        close to the surface, which improves the quality of the occlusion.

        :param kernel_size: Number of sample vectors to generate.
        :return: List of 3-component numpy arrays.
        """
        self.kernel = []
        for i in range(kernel_size):
            sample = np.array(
                [
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(0.0, 1.0),
                ],
                dtype=np.float32,
            )
            sample = sample / np.linalg.norm(sample)
            scale = float(i) / kernel_size
            scale = 0.1 + 0.9 * (scale * scale)
            sample = sample * scale
            self.kernel.append(sample)
        return self.kernel

    # -----------------------
    # SSAO Noise Texture Generation
    # -----------------------
    # Functions to generate SSAO noise texture.
    # -----------------------

    def generate_ssao_noise(self) -> np.ndarray:
        """Generate a small 4×4 noise texture for SSAO.

        The noise vectors rotate the sampling hemisphere around the normal.
        The texture is tiled across the screen to introduce noise.

        :return: A (4,4,3) float32 array of random vectors.
        """
        self.noise = np.zeros((4, 4, 3), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                self.noise[i, j] = np.array(
                    [
                        random.uniform(-1.0, 1.0),
                        random.uniform(-1.0, 1.0),
                        0.0,
                    ],
                    dtype=np.float32,
                )
                self.noise[i, j] = self.noise[i, j] / np.linalg.norm(
                    self.noise[i, j]
                )  # normalize (Optional)
        return self.noise

    # -----------------------
    # SSAO Noise Texture Creation
    # -----------------------
    # Functions to create SSAO noise texture.
    # -----------------------

    def create_noise_texture(self, noise: np.ndarray) -> int:
        """Create an OpenGL texture from the SSAO noise data.

        :param noise: A (4,4,3) float32 array of random vectors.
        :return: OpenGL texture id.
        """
        noise_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, noise_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, 4, 4, 0, GL.GL_RGB, GL.GL_FLOAT, noise
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        return noise_texture

    # -----------------------
    # Frame Buffer Creation
    # -----------------------
    # Functions to create frame buffers for various rendering passes.
    # -----------------------

    def create_frame_buffers(self) -> None:
        """
        Allocates FBO and textures for SSAO, shadow mapping, and other passes.

        :param self: The Renderer instance
        """
        self.shadowsize = utils.shadow_cfg.get(
            "resolution"
        )  # High-res shadow map for detailed shadows, impacts performance very much
        self.depth_fbo, self.depth_texture = self.create_point_shadow_map(
            self.shadowsize
        )
        self.ssao_data = self.create_ssao_buffers(self.width, self.height)

    # -----------------------
    # Uniform Location Caching
    # -----------------------
    # Cache frequently used uniform locations for performance.
    # -----------------------

    def cache_uniform_locations(self) -> None:
        """Query and store frequently accessed uniform locations."""

        self.depth_model_loc = GL.glGetUniformLocation(self.depth_program, "model")
        self.depth_light_pos_loc = GL.glGetUniformLocation(
            self.depth_program, "lightPos"
        )
        self.depth_far_plane_loc = GL.glGetUniformLocation(
            self.depth_program, "far_plane"
        )
        self.g_model_loc = GL.glGetUniformLocation(self.geometry_program, "model")
        self.g_view_loc = GL.glGetUniformLocation(self.geometry_program, "view")
        self.g_proj_loc = GL.glGetUniformLocation(self.geometry_program, "projection")
        self.g_object_color_loc = GL.glGetUniformLocation(
            self.geometry_program, "objectColor"
        )
        self.final_is_skinned_loc = GL.glGetUniformLocation(
            self.final_program, "u_is_skinned"
        )
        self.final_model_loc = GL.glGetUniformLocation(self.final_program, "model")
        self.final_view_loc = GL.glGetUniformLocation(self.final_program, "view")
        self.final_proj_loc = GL.glGetUniformLocation(self.final_program, "projection")
        self.final_lightspace_loc = GL.glGetUniformLocation(
            self.final_program, "lightSpaceMatrix"
        )
        self.final_light_pos_loc = GL.glGetUniformLocation(
            self.final_program, "lightPos"
        )
        self.final_view_pos_loc = GL.glGetUniformLocation(self.final_program, "viewPos")
        self.final_light_color_loc = GL.glGetUniformLocation(
            self.final_program, "lightColor"
        )
        self.final_light_intensity_loc = GL.glGetUniformLocation(
            self.final_program, "u_lightIntensity"
        )
        self.final_ambient_strength_loc = GL.glGetUniformLocation(
            self.final_program, "u_ambientStrength"
        )
        self.final_specular_strength_loc = GL.glGetUniformLocation(
            self.final_program, "u_specularStrength"
        )
        self.final_shininess_loc = GL.glGetUniformLocation(
            self.final_program, "u_shininess"
        )
        self.final_object_color_loc = GL.glGetUniformLocation(
            self.final_program, "objectColor"
        )
        self.final_reflection_vp_loc = GL.glGetUniformLocation(
            self.final_program, "reflectionViewProj"
        )
        self.final_bones_loc = GL.glGetUniformLocation(self.final_program, "u_bones")
        self.final_soft_shadows_loc = GL.glGetUniformLocation(
            self.final_program, "u_softShadows"
        )
        self.final_pcf_samples_loc = GL.glGetUniformLocation(
            self.final_program, "u_pcfSamples"
        )
        self.final_pcf_radius_loc = GL.glGetUniformLocation(
            self.final_program, "u_pcfRadius"
        )
        self.final_double_sided_loc = GL.glGetUniformLocation(
            self.final_program, "u_double_sided"
        )
        self.final_opacity_loc = GL.glGetUniformLocation(
            self.final_program, "u_opacity"
        )
        self.final_texture_loc = GL.glGetUniformLocation(
            self.final_program, "u_texture"
        )
        self.final_use_texture_loc = GL.glGetUniformLocation(
            self.final_program, "u_use_texture"
        )
        self.final_texture_mode_loc = GL.glGetUniformLocation(
            self.final_program, "u_texture_mode"
        )
        self.final_triplanar_scale_loc = GL.glGetUniformLocation(
            self.final_program, "u_triplanar_scale"
        )
        self.final_far_plane_loc = GL.glGetUniformLocation(
            self.final_program, "far_plane"
        )
        self.final_depth_map_loc = GL.glGetUniformLocation(
            self.final_program, "depthMap"
        )
        self.final_ssao_texture_loc = GL.glGetUniformLocation(
            self.final_program, "ssaoTexture"
        )
        self.ssao_gpos_loc = GL.glGetUniformLocation(self.ssao_program, "gPosition")
        self.ssao_gnormal_loc = GL.glGetUniformLocation(self.ssao_program, "gNormal")
        self.ssao_noise_loc = GL.glGetUniformLocation(self.ssao_program, "noiseTex")
        self.ssao_blur_input_loc = GL.glGetUniformLocation(
            self.ssao_blur_program, "ssaoInput"
        )

    # -----------------------
    # Shadow mapping
    # -----------------------
    # Functions related to shadow mapping for point lights.
    # -----------------------

    def point_light_matrices(
        self, light_pos=None, near_plane=None, far_plane=None
    ) -> list:
        """
        Generate view-projection matrices for omnidirectional
        point-light shadow mapping.

        Creates six matrices (cubemap faces).

        :param light_pos: Optional override light position
        :param near_plane: Near clip distance
        :param far_plane: Far clip distance
        :return: List of 6 shadow matrices
        """
        if light_pos is None:
            light_pos = self.light_pos
        if near_plane is None:
            near_plane = utils.shadow_cfg.get("near_plane")
        if far_plane is None:
            far_plane = utils.shadow_cfg.get("far_plane")

        proj = self.camera.get_projection_matrix(
            aspect=1.0, near=near_plane, far=far_plane, fov=90.0
        )

        matrices = []

        matrices.append(
            proj
            @ look_at(light_pos, light_pos + np.array([1, 0, 0]), np.array([0, -1, 0]))
        )
        matrices.append(
            proj
            @ look_at(light_pos, light_pos + np.array([-1, 0, 0]), np.array([0, -1, 0]))
        )
        matrices.append(
            proj
            @ look_at(light_pos, light_pos + np.array([0, 1, 0]), np.array([0, 0, 1]))
        )
        matrices.append(
            proj
            @ look_at(light_pos, light_pos + np.array([0, -1, 0]), np.array([0, 0, -1]))
        )
        matrices.append(
            proj
            @ look_at(light_pos, light_pos + np.array([0, 0, 1]), np.array([0, -1, 0]))
        )
        matrices.append(
            proj
            @ look_at(light_pos, light_pos + np.array([0, 0, -1]), np.array([0, -1, 0]))
        )

        return matrices

    # -----------------------
    # Shadow pass
    # -----------------------
    # This pass renders the scene from the light's perspective to create a
    # depth cubemap for shadow mapping.
    # -----------------------

    def render_shadow_pass(self, scene_objects, avatars=None) -> None:
        """
        Shadow pass.

        Renders scene depth from the light's perspective into
        a cubemap depth texture for point-light shadow mapping.

        :param scene_objects: Renderable scene objects
        :param avatars: Optional animated models
        """
        GL.glViewport(0, 0, self.shadowsize, self.shadowsize)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_fbo)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_FRONT)

        GL.glUseProgram(self.depth_program)

        # --- Point light shadow matrices ---
        shadow_mats = self.point_light_matrices()

        for i, mat in enumerate(shadow_mats):
            loc = GL.glGetUniformLocation(self.depth_program, f"shadowMatrices[{i}]")
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat)

        GL.glUniform3fv(self.depth_light_pos_loc, 1, self.light_pos)
        GL.glUniform1f(self.depth_far_plane_loc, utils.shadow_cfg.get("far_plane"))

        if avatars:
            for avatar in avatars:
                GL.glUniformMatrix4fv(
                    self.depth_model_loc, 1, GL.GL_TRUE, avatar.matrix()
                )
                avatar.mesh.draw()

        for obj in scene_objects:
            GL.glUniformMatrix4fv(
                self.depth_model_loc,
                1,
                GL.GL_TRUE,
                obj.transform.matrix(),
            )
            if getattr(obj.material, "double_sided", False):
                # For thin geometry like cloth:
                # Render BOTH sides into shadow map
                GL.glDisable(GL.GL_CULL_FACE)
            else:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_FRONT)

            obj.mesh.draw()

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # -----------------------
    # Fullscreen Quad Creation
    # -----------------------
    # Functions to create a fullscreen quad for post-processing effects.
    # -----------------------

    def create_fullscreen_quad(self):
        """
        Docstring für create_fullscreen_quad

        :param self: The object itself
        description: Creates a fullscreen quad for post-processing effects.
        """
        # fmt: off
        vertices = np.array([
            # pos      # uv
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            1.0,  1.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
            1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype=np.float32)
        # fmt: on

        self.quad_vao = GL.glGenVertexArrays(1)
        self.quad_vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )

        stride = 4 * 4  # 4 floats per vertex

        # position (location = 0)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0)
        )

        # texcoord (location = 1)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(8)
        )

        GL.glBindVertexArray(0)

    # -----------------------
    # SSAO pass
    # -----------------------
    # This pass computes Screen Space Ambient Occlusion (SSAO) for the scene.
    # It involves rendering the scene to a G-buffer, evaluating SSAO, and blurring
    # the result to reduce noise.
    # -----------------------

    def render_ssao_pass(self, camera, scene_objects: list[RenderObject]) -> None:
        """
        Screen Space Ambient Occlusion pass.

        Pipeline:
        1. Geometry pass → write position/normal to G-buffer
        2. SSAO evaluation → hemisphere sampling
        3. Blur pass

        :param camera: Active camera
        :param scene_objects: Scene geometry
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_data["g_fbo"])
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(int(GL.GL_COLOR_BUFFER_BIT) | int(GL.GL_DEPTH_BUFFER_BIT))
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self.geometry_program)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(self.width / self.height)

        GL.glUniformMatrix4fv(self.g_view_loc, 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(self.g_proj_loc, 1, GL.GL_TRUE, proj)

        for obj in scene_objects:
            GL.glUniformMatrix4fv(
                self.g_model_loc, 1, GL.GL_TRUE, obj.transform.matrix()
            )

            GL.glUniform3f(self.g_object_color_loc, 1.0, 1.0, 1.0)

            if getattr(obj.material, "double_sided", False):
                GL.glDisable(GL.GL_CULL_FACE)
            else:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_BACK)

            obj.mesh.draw()

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_data["ssao_fbo"])
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self.ssao_program)

        # G-buffer inputs
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_data["g_position"])
        GL.glUniform1i(self.ssao_gpos_loc, 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_data["g_normal"])
        GL.glUniform1i(self.ssao_gnormal_loc, 1)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_noise_texture)
        GL.glUniform1i(self.ssao_noise_loc, 2)

        # fullscreen quad
        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_data["ssao_blur_fbo"])
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.ssao_blur_program)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_data["ssao_tex"])
        GL.glUniform1i(self.ssao_blur_input_loc, 0)

        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # -----------------------
    # Bloom pass
    # -----------------------
    # This pass applies a bloom effect to bright areas of the scene.
    # It consists of three stages: bright pass, blur, and final combination.
    # -----------------------

    def render_bloom_pass(self) -> None:
        bloom = utils.bloom_cfg
        if not bloom.get("enabled", False):
            return

        threshold = bloom.get(
            "threshold"
        )  # brightness threshold, determines what is "bright"
        intensity = bloom.get(
            "intensity"
        )  # bloom intensity, determines how strong the bloom effect is
        blur_passes = 6  # number of blur passes, determines how much the bloom spreads

        """
        Bloom post-processing pass.

        Steps:
        - Bright extraction
        - Multi-pass Gaussian blur
        - Final HDR composition
        """
        # ---------- Bright pass ----------
        GL.glUseProgram(self.bloom_bright_program)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.blur_fbo[0])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hdr_color)
        GL.glUniform1i(
            GL.glGetUniformLocation(self.bloom_bright_program, "hdrScene"), 0
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.bloom_bright_program, "threshold"), threshold
        )

        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        # ---------- Blur ----------
        GL.glUseProgram(self.bloom_blur_program)

        horizontal = True
        read_tex = 0
        write_fbo = 1

        for _ in range(blur_passes):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.blur_fbo[write_fbo])
            GL.glUniform1i(
                GL.glGetUniformLocation(self.bloom_blur_program, "horizontal"),
                horizontal,
            )

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.blur_tex[read_tex])
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

            horizontal = not horizontal
            read_tex, write_fbo = write_fbo, read_tex

        final_blur_tex = self.blur_tex[read_tex]

        # ---------- Final ----------
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glClear(int(GL.GL_COLOR_BUFFER_BIT) | int(GL.GL_DEPTH_BUFFER_BIT))

        GL.glUseProgram(self.bloom_final_program)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hdr_color)
        GL.glUniform1i(GL.glGetUniformLocation(self.bloom_final_program, "hdrScene"), 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, final_blur_tex)
        GL.glUniform1i(
            GL.glGetUniformLocation(self.bloom_final_program, "bloomBlur"), 1
        )

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.volumetric_tex)
        GL.glUniform1i(
            GL.glGetUniformLocation(self.bloom_final_program, "volumetricTex"), 2
        )

        GL.glUniform1f(
            GL.glGetUniformLocation(self.bloom_final_program, "bloomIntensity"),
            intensity,
        )

        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

    # -----------------
    # Render Volumetric Lighting
    # -----------------
    # This pass renders volumetric lighting effects, such as light shafts or god rays.
    # It uses the depth information from the HDR pass to create light scattering effects.
    # -----------------

    def render_volumetric_pass(self, camera):
        """
        Volumetric lighting pass (light shafts / god rays).

        Uses shadow cubemap + scene depth to perform
        ray-marching in screen space.

        :param camera: Active camera
        """
        if not utils.volumetric_cfg.get("enabled", False):
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.volumetric_fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.volumetric_program)

        cfg = utils.volumetric_cfg

        GL.glUniform1i(
            GL.glGetUniformLocation(self.volumetric_program, "u_samples"),
            cfg.get("samples"),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.volumetric_program, "u_density"),
            cfg.get("density"),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.volumetric_program, "u_decay"),
            cfg.get("decay"),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.volumetric_program, "u_weight"),
            cfg.get("weight"),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.volumetric_program, "u_exposure"),
            cfg.get("exposure"),
        )

        GL.glUniform3fv(
            GL.glGetUniformLocation(self.volumetric_program, "lightPos"),
            1,
            self.light_pos,
        )

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.volumetric_program, "projection"),
            1,
            GL.GL_TRUE,
            self.camera.get_projection_matrix(self.width / self.height),
        )

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.volumetric_program, "view"),
            1,
            GL.GL_TRUE,
            camera.get_view_matrix(),
        )

        # Use actual camera world position (from player)
        GL.glUniform3fv(
            GL.glGetUniformLocation(self.volumetric_program, "viewPos"),
            1,
            camera.player.position,
        )

        GL.glUniform1f(
            GL.glGetUniformLocation(self.volumetric_program, "far_plane"),
            utils.shadow_cfg.get("far_plane"),
        )

        # --- Bind depth cubemap as depthMap at texture unit 0 ---
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.depth_texture)
        GL.glUniform1i(GL.glGetUniformLocation(self.volumetric_program, "depthMap"), 0)

        # --- Bind scene depth texture for geometry ray stop ---
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hdr_depth)
        GL.glUniform1i(
            GL.glGetUniformLocation(self.volumetric_program, "sceneDepth"), 1
        )

        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ----------------
    # Render Mannequin
    # ----------------
    # Here we render the player mannequin separately to avoid SSAO and shadows on it.
    # This ensures the mannequin is always clearly visible to the player.
    # It is drawn after the main scene rendering.
    # ----------------

    def render_player(self, mannequin):
        """
        Render player mannequin in final pass.

        Drawn after main scene to avoid unwanted SSAO/shadow
        artifacts on the player model.
        """

        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.final_program)

        # Mannequin wird aktuell NICHT geskinnt
        GL.glUniform1i(self.final_is_skinned_loc, 0)

        GL.glUniformMatrix4fv(self.final_model_loc, 1, GL.GL_TRUE, mannequin.matrix())

        GL.glUniform3f(self.final_object_color_loc, *mannequin.material.color)

        # --- BONES ---
        bones = mannequin.skeleton.bone_matrices
        GL.glUniformMatrix4fv(self.final_bones_loc, bones.shape[0], GL.GL_FALSE, bones)

        if mannequin.material.texture is not None:
            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mannequin.material.texture)
            GL.glUniform1i(self.final_texture_loc, 3)
            GL.glUniform1i(self.final_use_texture_loc, 1)
        else:
            GL.glUniform1i(self.final_use_texture_loc, 0)

        GL.glUniform1f(
            self.final_specular_strength_loc, mannequin.material.specular_strength
        )
        GL.glUniform1f(self.final_shininess_loc, mannequin.material.shininess)
        print("Bone[0] matrix:\n", mannequin.skeleton.bone_matrices[0])
        mannequin.mesh.draw()

        GL.glEnable(GL.GL_CULL_FACE)

    # -----------------------
    # Final pass
    # -----------------------
    # This pass combines all previous passes and applies lighting
    # and shading to produce the final image.
    # -----------------------

    def render_final_pass(
        self,
        mannequin,
        player,
        camera,
        scene_objects: list[RenderObject],
        screen_width,
        screen_height,
        debug_renderer=None,
    ) -> None:
        """
        Final lighting composition pass.

        Combines:
        - Shadow mapping
        - SSAO
        - HDR
        - Transparency
        - Optional debug grid

        Performs:
        - Opaque pass
        - Transparent sorted pass
        - Player rendering

        :param mannequin: Player model
        :param player: Player controller
        :param camera: Active camera
        :param scene_objects: Renderable objects
        :param screen_width: Framebuffer width
        :param screen_height: Framebuffer height
        :param debug_renderer: Optional debug renderer
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.hdr_fbo)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glViewport(0, 0, screen_width, screen_height)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(int(GL.GL_COLOR_BUFFER_BIT) | int(GL.GL_DEPTH_BUFFER_BIT))

        GL.glUseProgram(self.final_program)

        # =========================
        # Global uniforms
        # =========================
        GL.glUniformMatrix4fv(
            self.final_proj_loc,
            1,
            GL.GL_TRUE,
            self.camera.get_projection_matrix(self.width / self.height),
        )
        GL.glUniformMatrix4fv(
            self.final_view_loc, 1, GL.GL_TRUE, camera.get_view_matrix()
        )

        # =========================
        # Light uniforms
        # =========================
        GL.glUniform3fv(self.final_light_pos_loc, 1, self.light_pos)
        GL.glUniform3fv(self.final_view_pos_loc, 1, player.position)
        GL.glUniform3fv(self.final_light_color_loc, 1, self.light_color)

        GL.glUniform1f(self.final_light_intensity_loc, self.light_intensity)
        GL.glUniform1f(self.final_ambient_strength_loc, self.light_ambient)

        GL.glUniform1f(
            self.final_far_plane_loc,
            utils.shadow_cfg.get("far_plane"),
        )

        GL.glUniform1i(
            self.final_soft_shadows_loc,
            1 if utils.shadow_cfg.get("soft_shadows") else 0,
        )

        GL.glUniform1i(self.final_pcf_samples_loc, utils.shadow_cfg.get("pcf_samples"))

        GL.glUniform1f(self.final_pcf_radius_loc, utils.shadow_cfg.get("pcf_radius"))

        # =========================
        # Bind Textures
        # =========================
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.depth_texture)
        GL.glUniform1i(self.final_depth_map_loc, 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_data["ssao_blur_texture"])
        GL.glUniform1i(self.final_ssao_texture_loc, 1)

        # =========================
        # Split opaque / transparent
        # =========================
        opaque = []
        transparent = []

        for obj in scene_objects:
            if getattr(obj.material, "opacity", 1.0) < 1.0:
                transparent.append(obj)
            else:
                opaque.append(obj)

        # =========================
        # OPAQUE PASS
        # =========================
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)

        for obj in opaque:

            GL.glUniformMatrix4fv(
                self.final_model_loc,
                1,
                GL.GL_TRUE,
                obj.transform.matrix(),
            )

            GL.glUniform3f(self.final_object_color_loc, *obj.material.color)

            GL.glUniform1f(
                self.final_specular_strength_loc, obj.material.specular_strength
            )

            GL.glUniform1f(self.final_shininess_loc, obj.material.shininess)

            GL.glUniform1f(
                self.final_opacity_loc, getattr(obj.material, "opacity", 1.0)
            )

            GL.glUniform1i(
                self.final_double_sided_loc,
                1 if getattr(obj.material, "double_sided", False) else 0,
            )

            # Texture binding
            if obj.material.texture is not None:
                GL.glActiveTexture(GL.GL_TEXTURE3)
                GL.glBindTexture(GL.GL_TEXTURE_2D, obj.material.texture)
                GL.glUniform1i(self.final_texture_loc, 3)
                GL.glUniform1i(self.final_use_texture_loc, 1)
            else:
                GL.glUniform1i(self.final_use_texture_loc, 0)

            # Triplanar
            mode = getattr(obj.material, "texture_scale_mode", "default")

            if mode == "default":
                GL.glUniform1i(self.final_texture_mode_loc, 0)
                GL.glUniform1f(self.final_triplanar_scale_loc, 1.0)

            elif mode == "triplanar":
                GL.glUniform1i(self.final_texture_mode_loc, 1)
                GL.glUniform1f(
                    self.final_triplanar_scale_loc,
                    getattr(obj.material, "texture_scale_value"),
                )

            # Culling
            if getattr(obj.material, "double_sided", False):
                GL.glDisable(GL.GL_CULL_FACE)
            else:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_BACK)

            obj.mesh.draw()

        # =========================
        # TRANSPARENT PASS
        # =========================
        if transparent:

            cam_pos = player.position

            transparent.sort(
                key=lambda o: np.linalg.norm(o.transform.position - cam_pos),
                reverse=True,
            )

            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)

            for obj in transparent:

                GL.glUniformMatrix4fv(
                    self.final_model_loc,
                    1,
                    GL.GL_TRUE,
                    obj.transform.matrix(),
                )

                GL.glUniform3f(self.final_object_color_loc, *obj.material.color)

                GL.glUniform1f(
                    self.final_specular_strength_loc, obj.material.specular_strength
                )

                GL.glUniform1f(self.final_shininess_loc, obj.material.shininess)

                GL.glUniform1f(
                    self.final_opacity_loc, getattr(obj.material, "opacity", 1.0)
                )

                GL.glUniform1i(
                    self.final_double_sided_loc,
                    1 if getattr(obj.material, "double_sided", False) else 0,
                )

                if obj.material.texture is not None:
                    GL.glActiveTexture(GL.GL_TEXTURE3)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, obj.material.texture)
                    GL.glUniform1i(self.final_texture_loc, 3)
                    GL.glUniform1i(self.final_use_texture_loc, 1)
                else:
                    GL.glUniform1i(self.final_use_texture_loc, 0)

                # Triplanar
                mode = getattr(obj.material, "texture_scale_mode", "default")

                if mode == "default":
                    GL.glUniform1i(self.final_texture_mode_loc, 0)
                    GL.glUniform1f(self.final_triplanar_scale_loc, 1.0)

                elif mode == "triplanar":
                    GL.glUniform1i(self.final_texture_mode_loc, 1)
                    GL.glUniform1f(
                        self.final_triplanar_scale_loc,
                        getattr(obj.material, "texture_scale_value"),
                    )

                if getattr(obj.material, "double_sided", False):
                    GL.glDisable(GL.GL_CULL_FACE)
                else:
                    GL.glEnable(GL.GL_CULL_FACE)
                    GL.glCullFace(GL.GL_BACK)

                obj.mesh.draw()

            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)

        # =========================
        # Draw player mannequin
        # =========================
        if mannequin is not None:
            self.render_player(mannequin)

        # =========================
        # Debug Grid (3D world space inside HDR FBO)
        # =========================
        if debug_renderer is not None:
            debug_renderer.draw_debug_grid(
                camera, screen_width / screen_height, size=100.0
            )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
