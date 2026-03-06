import pygame
from OpenGL import GL


class Texture:

    @staticmethod
    def load_texture(path: str) -> int:
        """
        Load a PNG/JPG texture from disk and upload it to OpenGL.
        Returns the OpenGL texture ID.
        """

        # Load image via pygame
        surface = pygame.image.load(path).convert_alpha()
        width, height = surface.get_size()

        # Convert to raw bytes (RGBA)
        image_data = pygame.image.tostring(surface, "RGBA", True)

        # Create OpenGL texture
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

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

        # Texture filtering
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        # Generate mipmaps
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        return tex_id
