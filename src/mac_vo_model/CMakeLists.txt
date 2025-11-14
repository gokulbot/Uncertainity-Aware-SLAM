cmake_minimum_required(VERSION 3.10)
project(mac_vo_model)

####### mac vo model library ##########

# Add libraries and linking files for imu pre integration module gtsam 
add_library(mac_vo_model SHARED src/model.cpp)

# include directories for the library
target_include_directories(mac_vo_model PUBLIC ${FRAMES_INCLUDE_DIRS} include)

# linking the required libraries
target_link_libraries(mac_vo_model PUBLIC ${OpenCV_LIBS} timer logger torch)

# test executable
add_executable(test_mac_vo_model tests/test_mac_vo_model.cpp)
target_link_libraries(test_mac_vo_model PRIVATE mac_vo_model ${OpenCV_LIBS} timer logger torch)