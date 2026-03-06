#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
uniform vec2 u_offset; // pixel offset
uniform vec2 u_scale;  // pixel scale
uniform vec2 u_view;   // viewport size
void main() {
  // convert from pixel space to NDC
  vec2 pos = aPos * u_scale + u_offset;
  vec2 ndc = (pos / u_view) * 2.0 - 1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
  vUV = aUV;
}