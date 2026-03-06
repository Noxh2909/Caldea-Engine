#version 330 core

layout(location = 0) in vec3 aPos;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;

out vec3 v_world_pos;

void main() {
  vec4 world = u_model * vec4(aPos, 1.0);
  v_world_pos = world.xyz;
  gl_Position = u_proj * u_view * world;
}