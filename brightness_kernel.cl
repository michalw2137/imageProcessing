__kernel void brightness(__global uchar* input, __global uchar* output, int width, int height, int brightnessValue)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = y * width + x;

    if (x < width && y < height)
    {
        uchar pixel = input[index];
        output[index] = clamp(pixel + brightnessValue, 0, 255);
    }
}
