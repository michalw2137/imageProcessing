#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <CL/cl2.hpp>

std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel source file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    try {
        // Read input image
        cv::Mat inputImage = cv::imread("../input.png");
        if (inputImage.empty()) {
            throw std::runtime_error("Failed to load input image!");
        }
        std::cout << "Input image loaded successfully." << std::endl;

        // Convert input image to grayscale
        cv::Mat grayscaleImage;
        cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

        // Save grayscale image
        cv::imwrite("../grey.png", grayscaleImage);
        std::cout << "Grayscale image conversion completed." << std::endl;

        // Get image dimensions
        int width = grayscaleImage.cols;
        int height = grayscaleImage.rows;

        // Load kernel source code
        std::string kernelSource = readKernelSource("../sobel_kernel.cl");
        std::cout << "Kernel source code loaded successfully." << std::endl;

        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // Select the first platform
        cl::Platform platform = platforms[0];
        std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // Get available devices on the selected platform
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found on platform");
        }

        // Select the first device
        cl::Device device = devices[0];
        std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create a context for the selected device
        cl::Context context(device);

        // Create a command queue
        cl::CommandQueue queue(context, device);

        // Create the program from the kernel source code
        cl::Program program(context, kernelSource);

        // Build the program for the selected device
        try {
            program.build({device});
        } catch (...) {
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            throw std::runtime_error("Failed to build OpenCL program: " + log);
        }
        std::cout << "OpenCL program built successfully." << std::endl;

        // Create an input buffer
        cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(uchar), grayscaleImage.data);

        // Create an output buffer
        cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(uchar));

        // Create a kernel
        cl::Kernel kernel(program, "sobel");

        // Set kernel arguments
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, width);
        kernel.setArg(3, height);

        // Enqueue the kernel for execution
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
        std::cout << "OpenCL kernel enqueued for execution." << std::endl;

        // Read the output buffer back to the host
        std::vector<uchar> outputData(width * height);
        queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, width * height * sizeof(uchar), outputData.data());
        std::cout << "Output data read back to host." << std::endl;

        // Create the output image
        cv::Mat outputImage(height, width, CV_8UC1, outputData.data());

        // Save the output image
        cv::imwrite("../output.png", outputImage);
        std::cout << "Sobel edge detection result saved as output.png" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "An error occurred: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
