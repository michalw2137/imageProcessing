#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl2.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // Load the kernel source code from a file
    std::ifstream kernelFile("../brightness_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file." << std::endl;
        return 1;
    }

    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());

    // Get available OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    // Log available platforms
    std::cout << "Available OpenCL Platforms:" << std::endl;
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    // Select the first platform
    cl::Platform platform = platforms[0];

    // Get available devices for the platform
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found." << std::endl;
        return 1;
    }

    // Log available devices
    std::cout << "Available OpenCL Devices:" << std::endl;
    for (const auto& device : devices) {
        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    // Select the first device
    cl::Device device = devices[0];

    // Create an OpenCL context for the selected device
    cl::Context context({device});

    // Create a command queue
    cl::CommandQueue queue(context, device);

    // Create a program from the kernel source code
    cl::Program program(context, kernelSource);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "Failed to build the OpenCL program." << std::endl;
        std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    // Load the input image using OpenCV
    cv::Mat inputImage = cv::imread("../input.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Failed to load the input image." << std::endl;
        return 1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Create input and output buffers on the device
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * width * height, inputImage.data);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * width * height);

    // Create a kernel object
    cl::Kernel kernel(program, "sobel");

    float threshholdX = 1.0f;
    float threshholdY = 1.0f;
    // Set kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, threshholdX);
    kernel.setArg(5, threshholdY);

    // Enqueue the kernel for execution
    cl::NDRange globalSize(width, height);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);

    // Read the output image data from the device to the host
    std::vector<uchar> outputImage(width * height);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(uchar) * width * height, outputImage.data());

    // Create a cv::Mat object from the output image data
    cv::Mat outputImageMat(height, width, CV_8UC1, outputImage.data());

    // Save the output image using OpenCV
    cv::imwrite("../output.png", outputImageMat);

    // Print information about the output image
    std::cout << "Output Image Information:" << std::endl;
    std::cout << "Rows: " << outputImageMat.rows << std::endl;
    std::cout << "Columns: " << outputImageMat.cols << std::endl;
    std::cout << "Min Pixel Value: " << static_cast<int>(*std::min_element(outputImageMat.begin<uchar>(), outputImageMat.end<uchar>())) << std::endl;
    std::cout << "Max Pixel Value: " << static_cast<int>(*std::max_element(outputImageMat.begin<uchar>(), outputImageMat.end<uchar>())) << std::endl;

    std::cout << "Sobel edge detection completed." << std::endl;

    return 0;
}
