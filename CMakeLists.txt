cmake_minimum_required(VERSION 3.5)

project(MAC_VO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(torch_DIR "/opt/libtorch/share/cmake/Torch")    

################################################find packages###############################################
find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(torch REQUIRED)

############################################include directories###############################################
set(OpenCV_DIR /usr/local/opencv4/lib/cmake/opencv4)
set(EIGEN_DIR /usr/include/eigen3/)
set(FRAMES_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/include)

##########################################libraries###############################################
add_subdirectory(src/mac_vo_model)  
add_subdirectory(src/utils)
add_subdirectory(src/logger)
add_subdirectory(src/timer)

#######################################executables###############################################



