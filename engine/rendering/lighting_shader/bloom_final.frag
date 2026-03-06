#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D hdrScene;
uniform sampler2D bloomBlur;
uniform sampler2D volumetricTex;
uniform float bloomIntensity;

void main()
{
    vec3 hdr = texture(hdrScene, TexCoord).rgb;
    vec3 bloom = texture(bloomBlur, TexCoord).rgb;
    vec3 volumetric = texture(volumetricTex, TexCoord).rgb;

    // Base HDR + Bloom
    vec3 color = hdr + bloom * bloomIntensity;

    // Add volumetric lighting (god rays)
    color += volumetric;

    FragColor = vec4(color, 1.0);
}