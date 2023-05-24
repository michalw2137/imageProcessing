__kernel void sobel(__global const uchar* inputImage, __global uchar* outputImage, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -inputImage[(y - 1) * width + (x - 1)] + inputImage[(y - 1) * width + (x + 1)]
                 - 2 * inputImage[y * width + (x - 1)] + 2 * inputImage[y * width + (x + 1)]
                 - inputImage[(y + 1) * width + (x - 1)] + inputImage[(y + 1) * width + (x + 1)];

        int gy = -inputImage[(y - 1) * width + (x - 1)] - 2 * inputImage[(y - 1) * width + x] - inputImage[(y - 1) * width + (x + 1)]
                 + inputImage[(y + 1) * width + (x - 1)] + 2 * inputImage[(y + 1) * width + x] + inputImage[(y + 1) * width + (x + 1)];

        int magnitude = (int)sqrt(gx * gx + gy * gy);
        outputImage[y * width + x] = (uchar)(magnitude > 255 ? 255 : magnitude);
    } else {
        // Pixels on the edge of the image are set to 0 directly
        outputImage[y * width + x] = 0;
    }
}
