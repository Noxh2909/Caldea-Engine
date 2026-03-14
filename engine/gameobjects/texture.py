import pygame
from OpenGL import GL


class Texture:

    # -------------------------------------------------
    # Internal uploader (shared by all methods)
    # -------------------------------------------------
    @staticmethod
    def _upload_surface(surface) -> int:
        width, height = surface.get_size()

        image_data = pygame.image.tostring(surface, "RGBA", False)

        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            image_data,
        )

        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        return tex_id

    # -------------------------------------------------
    # Load from file (PNG/JPG)
    # -------------------------------------------------
    @staticmethod
    def load_texture(path: str) -> int:
        surface = pygame.image.load(path).convert_alpha()
        return Texture._upload_surface(surface)

    # -------------------------------------------------
    # Load from pygame surface (GLB usage)
    # -------------------------------------------------
    @staticmethod
    def load_texture_from_surface(surface) -> int:
        return Texture._upload_surface(surface)

    # -------------------------------------------------
    # Load directly from raw bytes (optional)
    # -------------------------------------------------
    @staticmethod
    def load_texture_from_bytes(image_bytes, size) -> int:
        surface = pygame.image.frombuffer(image_bytes, size, "RGBA")
        return Texture._upload_surface(surface)