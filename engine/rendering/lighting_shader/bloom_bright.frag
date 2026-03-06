#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D hdrScene;
uniform float threshold;

void main() {
  vec3 color = texture(hdrScene, TexCoord).rgb;
  float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
  FragColor = brightness > threshold ? vec4(color, 1.0) : vec4(0.0);
}