cmake_minimum_required(VERSION 3.25)
project(openCL)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_GENERATOR_PLATFORM x64)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ustaw ścieżki do include i biblioteki OpenCL
set(OpenCL_INCLUDE_DIR "C:/Users/Misza/.clion-vcpkg/vcpkg/packages/opencl_x64-windows/include")
set(OpenCL_LIBRARY "C:/Users/Misza/.clion-vcpkg/vcpkg/packages/opencl_x64-windows/lib/OpenCL.lib")

include_directories(${OpenCL_INCLUDE_DIR})

add_executable(openCL main.cpp)
target_link_libraries(openCL ${OpenCV_LIBS} ${OpenCL_LIBRARY})
