#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;

in vec3 WorldPos[];

out vec3 FragPos;

uniform mat4 shadowMatrices[6];

void main() {
    for (int face = 0; face < 6; ++face) {
        gl_Layer = face;

        for (int i = 0; i < 3; ++i) {
            FragPos = WorldPos[i];
            gl_Position = shadowMatrices[face] * vec4(WorldPos[i], 1.0);
            EmitVertex();
        }
        EndPrimitive();
    }
}