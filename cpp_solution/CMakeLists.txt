cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

project(llm_vlm_cpp_solution LANGUAGES CXX)

# tokenizer
set(TOKENZIER_CPP_PATH ${CMAKE_SOURCE_DIR}/tokenizers-cpp)
set(TOKENZIER_CPP_BINARY_DIR ${CMAKE_BINARY_DIR}/tokenizers-cpp)

# onnxruntime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.21.0")

file(MAKE_DIRECTORY ${TOKENZIER_CPP_BINARY_DIR}/release)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers-cpp EXCLUDE_FROM_ALL)

# solution
add_subdirectory(llm_solution)
add_subdirectory(vlm_solution)
