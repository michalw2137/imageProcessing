__kernel void sobel(__global uchar* input, __global uchar* output, int width, int height, float threshholdX, float threshholdY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int dx = -1 * input[(y - 1) * width + (x - 1)] + -2 * input[y * width + (x - 1)] + -1 * input[(y + 1) * width + (x - 1)] +
             input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

    int dy = -1 * input[(y - 1) * width + (x - 1)] + -2 * input[(y - 1) * width + x] + -1 * input[(y - 1) * width + (x + 1)] +
             input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    float dxFloat = convert_float(dx * threshholdX);
    float dyFloat = convert_float(dy * threshholdY);

    output[y * width + x] = convert_uchar(sqrt(dxFloat * dxFloat + dyFloat * dyFloat));
}
