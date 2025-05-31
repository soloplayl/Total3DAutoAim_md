cmake_minimum_required(VERSION 3.10)

project(infer_openvino)

# 添加库目标（名称必须合法）
add_library(infer_openvino 
    src/infer_openvino.cpp  # 源文件路径
    include/infer_openvino.hpp  # 头文件路径
)

# 包含头文件路径
target_include_directories(infer_openvino PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # 头文件所在目录
    ${OpenVINO_INCLUDE_DIRS}  # OpenVINO头文件
)

# 链接依赖项
target_link_libraries(infer_openvino PRIVATE
    openvino::runtime  # OpenVINO库
)