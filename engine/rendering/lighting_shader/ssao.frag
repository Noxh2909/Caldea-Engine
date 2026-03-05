#version 330 core
out float FragColor;
in vec2 TexCoord;
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D noiseTex;
uniform vec3 samples[64];
uniform mat4 projection;
uniform int kernelSize;
uniform float radius;
uniform float bias;
uniform vec2 noiseScale;
void main() {
  // Get view space position and normal
  vec3 fragPos = texture(gPosition, TexCoord).xyz;
  vec3 normal = normalize(texture(gNormal, TexCoord).xyz);
  // Retrieve a random vector from noise texture and rotate samples around
  // normal
  vec3 randomVec = texture(noiseTex, TexCoord * noiseScale).xyz;
  // Construct TBN matrix from normal and random vector
  vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
  vec3 bitangent = cross(normal, tangent);
  mat3 TBN = mat3(tangent, bitangent, normal);
  float occlusion = 0.0;
  for (int i = 0; i < kernelSize; ++i) {
    // Sample position in tangent space
    vec3 sampleVec = TBN * samples[i];
    sampleVec = fragPos + sampleVec * radius;
    // Project sample position (to sample texture)
    vec4 offset = vec4(sampleVec, 1.0);
    offset = projection * offset;
    offset.xyz /= offset.w;
    offset.xyz = offset.xyz * 0.5 + 0.5;
    float sampleDepth = texture(gPosition, offset.xy).z;
    float rangeCheck =
        smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
    if (sampleDepth >= sampleVec.z + bias) {
      occlusion += rangeCheck;
    }
  }
  occlusion = 1.0 - (occlusion / kernelSize);
  FragColor = occlusion;
}