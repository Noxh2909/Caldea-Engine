import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders


class DebugGizmo:
    def __init__(self):
        self.enabled = False

        vertex_src = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 uVP;
        void main() {
            gl_Position = uVP * vec4(aPos, 1.0);
        }
        """

        fragment_src = """
        #version 330 core
        out vec4 FragColor;
        uniform vec3 uColor;
        void main() {
            FragColor = vec4(uColor, 1.0);
        }
        """

        # Core profile requires a VAO bound during shader validation
        temp_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(temp_vao)

        self.shader = shaders.compileProgram(
            shaders.compileShader(vertex_src, GL.GL_VERTEX_SHADER),
            shaders.compileShader(fragment_src, GL.GL_FRAGMENT_SHADER),
        )
        GL.glBindVertexArray(0)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)

    def toggle(self):
        self.enabled = not self.enabled

    def draw_lines(self, vp_matrix, vertices, color=(0, 1, 0)):
        if not self.enabled:
            return

        # Render gizmos on top (disable depth test temporarily)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.shader)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)

        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vertices.astype(np.float32),
            GL.GL_DYNAMIC_DRAW
        )

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        # uniforms
        loc_vp = GL.glGetUniformLocation(self.shader, "uVP")
        loc_color = GL.glGetUniformLocation(self.shader, "uColor")

        # NumPy uses row-major matrices, OpenGL expects column-major.
        # Transpose before sending to shader.
        GL.glUniformMatrix4fv(loc_vp, 1, GL.GL_FALSE, vp_matrix.T)
        GL.glUniform3f(loc_color, *color)

        GL.glDrawArrays(GL.GL_LINES, 0, len(vertices))

        # Restore depth test
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)