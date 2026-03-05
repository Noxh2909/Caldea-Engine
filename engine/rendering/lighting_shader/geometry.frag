#version 330 core
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedo;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
uniform vec3 objectColor;
void main() {
  gPosition = FragPos;
  gNormal = normalize(Normal);
  gAlbedo = vec4(objectColor, 1.0);
}