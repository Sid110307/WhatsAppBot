cmake_minimum_required(VERSION 3.2)
project(WhatsAppBot)

set(CMAKE_CXX_STANDARD 20)

if (UNIX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "/W4")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(whatsAppBot
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/neuralNetwork.cpp
        ${PROJECT_SOURCE_DIR}/include/neuralNetwork.h)

target_compile_definitions(whatsAppBot PRIVATE USER="${USER}")
