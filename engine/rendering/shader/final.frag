#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
uniform sampler2D ssaoTexture;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform sampler2D u_texture;
uniform bool u_use_texture;
uniform bool u_is_emissive;
uniform bool u_softShadows;
uniform bool u_double_sided;
uniform int u_pcfSamples;
uniform float u_pcfRadius;
uniform int u_texture_mode; // 0 = UV, 1 = triplanar
uniform float u_lightIntensity;
uniform float u_ambientStrength;
uniform float u_specularStrength;
uniform float u_shininess;
uniform float u_triplanar_scale;
uniform samplerCube depthMap;
uniform vec3 lightPos;
uniform float far_plane;

// ----------------------
// Shadow calculation
// ----------------------

float ShadowCalculation(vec3 fragPos)
{
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight);

    // normal-based bias (reduces acne)
    vec3 N = normalize(Normal);
    if(u_double_sided && !gl_FrontFacing)
        N = -N;
    vec3 L = normalize(fragToLight);
    float bias = max(0.05 * (1.0 - dot(N, L)), 0.005);

    // simple PCF over cubemap
    float shadow = 0.0;
    int samples = u_pcfSamples;
    float diskRadius = u_pcfRadius;

    if (samples <= 0)
        return 0.0;

    for (int i = 0; i < samples; ++i)
    {
        // predefined offset directions (hardcoded pattern)
        vec3 offset = normalize(vec3(
            sin(float(i) * 12.9898),
            cos(float(i) * 78.233),
            sin(float(i) * 37.719)
        ));

        float closestDepth = texture(
            depthMap,
            fragToLight + offset * diskRadius
        ).r;

        closestDepth *= far_plane;

        if (currentDepth - bias > closestDepth)
            shadow += 1.0;
    }

    shadow /= float(samples);
    return shadow;
}

// Main function
void main() {

  // ----------------------
  // Texture sampling
  // ----------------------

  vec3 baseColor;

  if (u_use_texture) {
    if (u_texture_mode == 0) {
      // UV mapping
      baseColor = texture(u_texture, TexCoord).rgb;
    } else {
      // Triplanar mapping (world-space)
      vec3 n = normalize(Normal);
      vec3 blend = abs(n);
      blend = max(blend, vec3(0.0001));
      blend /= (blend.x + blend.y + blend.z);

      vec2 uvX = FragPos.zy * u_triplanar_scale;
      vec2 uvY = FragPos.xz * u_triplanar_scale;
      vec2 uvZ = FragPos.xy * u_triplanar_scale;

      vec3 tx = texture(u_texture, uvX).rgb;
      vec3 ty = texture(u_texture, uvY).rgb;
      vec3 tz = texture(u_texture, uvZ).rgb;

      baseColor = tx * blend.x + ty * blend.y + tz * blend.z;
    }
  } else {
    baseColor = objectColor;
  }

  // ----------------------
  // Lighting calculations
  // ----------------------

  vec2 screenUV = gl_FragCoord.xy / vec2(textureSize(ssaoTexture, 0));
  float ao = texture(ssaoTexture, screenUV).r;
  vec3 ambient = u_ambientStrength * ao * lightColor * u_lightIntensity;
  vec3 norm = normalize(Normal);
  if(u_double_sided && !gl_FrontFacing)
    norm = -norm;
  vec3 lightDir = normalize(lightPos - FragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor * baseColor * u_lightIntensity;
  vec3 viewDir = normalize(viewPos - FragPos);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_shininess);
  vec3 specular = spec * u_specularStrength * lightColor * u_lightIntensity;
  float shadow = ShadowCalculation(FragPos);

  // Add attenuation based on distance
  float distance = length(lightPos - FragPos);
  float attenuation =
      1.0 / (1.0 + 0.09 * distance +
             0.032 * distance * distance); // Quadratic attenuation

  vec3 lighting = ambient + attenuation * (1.0 - shadow) * (diffuse + specular);

  // ----------------------
  // Final output
  // ----------------------
  vec3 finalColor = lighting * baseColor;
  FragColor = vec4(finalColor, 1.0);
}