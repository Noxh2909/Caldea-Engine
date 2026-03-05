#version 330 core
    out float FragColor;
    in vec2 TexCoord;
    uniform sampler2D ssaoInput;
    uniform vec2 texelSize;
    void main() {
        float result = 0.0;
        float kernel[9] = float[](0.0625, 0.125, 0.0625,
                                  0.125,  0.25,  0.125,
                                  0.0625, 0.125, 0.0625);
        int index = 0;
        for(int y = -1; y <= 1; ++y) {
            for(int x = -1; x <= 1; ++x) {
                vec2 offset = TexCoord + vec2(float(x), float(y)) * texelSize;
                result += texture(ssaoInput, offset).r * kernel[index++];
            }
        }
        FragColor = result;
    }