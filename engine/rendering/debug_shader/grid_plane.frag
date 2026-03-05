#version 330 core

in vec3 v_world_pos;
out vec4 FragColor;

uniform vec3 u_color;

void main() {
  // simple grid-style coloring
  float scale = 1.0;
  float line = step(0.999, abs(sin(v_world_pos.x * scale))) +
               step(0.999, abs(sin(v_world_pos.z * scale)));

  vec3 base = vec3(0.15, 0.15, 0.17);

  // white grid lines
  vec3 grid_color = vec3(1.0, 1.0, 1.0);

  float alpha = clamp(line, 0.0, 1.0) * 0.25; // <<< grid transparency

  vec3 color = mix(base, grid_color, clamp(line, 0.0, 1.0));
  FragColor = vec4(color, alpha);
}