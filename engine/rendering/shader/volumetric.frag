#version 330 core

out vec4 FragColor;
in vec2 TexCoord;

uniform samplerCube depthMap;      // shadow cubemap
uniform sampler2D sceneDepth;      // scene depth (hdr_depth)

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform mat4 projection;
uniform mat4 view;

uniform float far_plane;

uniform int u_samples;
uniform float u_density;
uniform float u_decay;
uniform float u_weight;
uniform float u_exposure;

vec3 ReconstructWorldPosition(vec2 uv, float depth)
{
    float z = depth * 2.0 - 1.0;

    vec4 clip = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 viewPos4 = inverse(projection) * clip;
    viewPos4 /= viewPos4.w;

    vec4 worldPos4 = inverse(view) * viewPos4;
    return worldPos4.xyz;
}

void main()
{
    float depth = texture(sceneDepth, TexCoord).r;

    // if background pixel, skip
    if(depth >= 1.0)
    {
        FragColor = vec4(0.0);
        return;
    }

    vec3 surfaceWorldPos = ReconstructWorldPosition(TexCoord, depth);

    vec3 rayDir = normalize(surfaceWorldPos - viewPos);

    float rayLength = length(surfaceWorldPos - viewPos);
    float stepSize = rayLength / float(u_samples);

    vec3 currentPos = viewPos;

    // jitter to remove banding
    float noise = fract(sin(dot(TexCoord.xy ,vec2(12.9898,78.233))) * 43758.5453);
    currentPos += rayDir * noise * stepSize;

    float illumination = 0.0;
    float decay = 1.0;

    float lightRadius = 15.0;

    for(int i = 0; i < u_samples; i++)
    {
        currentPos += rayDir * stepSize;

        vec3 fragToLight = currentPos - lightPos;
        float currentDepth = length(fragToLight);

        if(currentDepth > lightRadius)
            continue;

        float closestDepth = texture(depthMap, fragToLight).r;
        closestDepth *= far_plane;

        float shadowBias = 0.02;
        float visibility = 0.0;

        if(currentDepth - shadowBias < closestDepth)
            visibility = 1.0;

        float attenuation = 1.0 / (1.0 + 0.09 * currentDepth + 0.032 * currentDepth * currentDepth);

        illumination += visibility * attenuation * decay * u_weight;

        decay *= u_decay;
    }

    float intensity = illumination * u_exposure;

    // small base fog so VL never fully disappears
    float baseFog = 0.02;
    intensity = baseFog + pow(intensity, 1.4);

    FragColor = vec4(vec3(intensity), 1.0);
}