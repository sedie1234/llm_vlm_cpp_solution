cmake_minimum_required(VERSION 3.16)

project(llm_cpp_solution LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# llm cpp solution
add_executable(llm_cpp_solution llm.cpp
                                ../utils/utils.cpp)
target_include_directories(llm_cpp_solution PRIVATE 
                            ${TOKENZIER_CPP_PATH}/include
                            ${ONNXRUNTIME_DIR}/include)

target_link_directories(llm_cpp_solution PRIVATE 
                            ${ONNXRUNTIME_DIR}/lib
                            ${CMAKE_BINARY_DIR}/tokenizers-cpp/release)

target_link_libraries(llm_cpp_solution PRIVATE 
                            tokenizers_cpp
                            onnxruntime)