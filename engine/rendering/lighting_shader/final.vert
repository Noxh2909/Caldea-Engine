#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in uvec4 aBoneIDs;
layout(location = 4) in vec4  aBoneWeights;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
#define MAX_BONES 128
uniform mat4 u_bones[MAX_BONES];
uniform bool u_is_skinned;
// ViewProjection matrix of the reflected camera (for sampling reflection
// texture)
uniform mat4 reflectionViewProj;
uniform mat4 lightSpaceMatrix;
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 FragPosLightSpace;
out vec4 ReflClipPos;
void main()
{
    // --- Skinning (explicit) ---
    mat4 skin = mat4(1.0);

    if (u_is_skinned)
    {
        skin =
              aBoneWeights.x * u_bones[int(aBoneIDs.x)] +
              aBoneWeights.y * u_bones[int(aBoneIDs.y)] +
              aBoneWeights.z * u_bones[int(aBoneIDs.z)] +
              aBoneWeights.w * u_bones[int(aBoneIDs.w)];
    }

    vec3 skinnedNormal = normalize(mat3(skin) * aNormal);
    vec4 skinnedPos    = skin * vec4(aPos, 1.0);

    // --- World space ---
    vec4 worldPos = model * skinnedPos;

    FragPos = worldPos.xyz;
    Normal  = mat3(transpose(inverse(model))) * skinnedNormal;
    TexCoord = aTexCoord;

    FragPosLightSpace = lightSpaceMatrix * worldPos;
    ReflClipPos       = reflectionViewProj * worldPos;

    gl_Position = projection * view * worldPos;
}