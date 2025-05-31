cmake_minimum_required(VERSION 3.10)

project(Total3DAutoAim_CPPInference VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# OpenCV
find_package(OpenCV REQUIRED)

find_package(OpenVINO REQUIRED)
# find_package(Ceres REQUIRED)
# # CUDA
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# find_package(CUDA 11 REQUIRED)

# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)


include_directories(${OpenCV_INCLUDE_DIRS})

# include_directories(${CERES_INCLUDE_DIRS})

# 添加子目录
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/infer_openvino)

# !OpenCV

set(PROJECT_SOURCES
    test.cpp
    )

add_executable(Total3DAutoAim_CPPInference ${PROJECT_SOURCES})


target_link_libraries(Total3DAutoAim_CPPInference 
        openvino::runtime
        infer_openvino
        # ${CERES_LIBRARIES}
        ${OpenCV_LIBS})