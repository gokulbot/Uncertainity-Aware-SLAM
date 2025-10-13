cmake_minimum_required(VERSION 3.10)
project(logger)

set(CMAKE_CXX_STANDARD 17)

find_package(spdlog REQUIRED)

add_library(logger SHARED src/logger.cpp)
target_include_directories(logger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${FRAMES_INCLUDE_DIRS})
target_link_libraries(logger spdlog::spdlog ${OpenCV_LIBS} )
