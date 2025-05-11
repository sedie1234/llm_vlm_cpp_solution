cmake_minimum_required(VERSION 3.16)

project(vlm_cpp_solution LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# onnxruntime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.21.0")

# find package
find_package(OpenCV REQUIRED)

# llm cpp solution
add_executable(vlm_cpp_solution vlm.cpp)
target_include_directories(vlm_cpp_solution PRIVATE 
                            ${TOKENZIER_CPP_PATH}/include
                            ${ONNXRUNTIME_DIR}/include)

target_link_directories(vlm_cpp_solution PRIVATE 
                            ${ONNXRUNTIME_DIR}/lib
                            ${CMAKE_BINARY_DIR}/tokenizers-cpp/release)

target_link_libraries(vlm_cpp_solution PRIVATE 
                            tokenizers_cpp
                            onnxruntime
                            ${OpenCV_LIBS})