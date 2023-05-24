__kernel void sobel(__global uchar* input, __global uchar* output, int width, int height, float threshholdX, float threshholdY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        output[y * width + x] = convert_uchar(0);
        return;
    }

    int dx = -1 * input[(y - 1) * width + (x - 1)] + -2 * input[y * width + (x - 1)] + -1 * input[(y + 1) * width + (x - 1)] +
             input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

    int dy = -1 * input[(y - 1) * width + (x - 1)] + -2 * input[(y - 1) * width + x] + -1 * input[(y - 1) * width + (x + 1)] +
             input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    float dxFloat = convert_float(dx * threshholdX);
    float dyFloat = convert_float(dy * threshholdY);

    float value = sqrt(dxFloat * dxFloat + dyFloat * dyFloat);
    if(value > 255) {
        output[y * width + x] = convert_uchar(255);
    } else {
        output[y * width + x] = convert_uchar(value);
    }
}
