#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// ViewProjection matrix of the reflected camera (for sampling reflection texture)
uniform mat4 reflectionViewProj;

// Shadow matrix
uniform mat4 lightSpaceMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 FragPosLightSpace;
out vec4 ReflClipPos;

void main()
{
    // --- World space ---
    vec4 worldPos = model * vec4(aPos, 1.0);

    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    // Shadow mapping
    FragPosLightSpace = lightSpaceMatrix * worldPos;

    // Planar reflections
    ReflClipPos = reflectionViewProj * worldPos;

    gl_Position = projection * view * worldPos;
}