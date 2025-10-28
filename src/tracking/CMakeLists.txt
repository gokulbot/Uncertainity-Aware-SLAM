cmake_minimum_required(VERSION 3.5)

project(tracking)

####################################################################################

add_library(tracking SHARED src/tracking.cpp)

target_include_directories(tracking PUBLIC include)
target_link_libraries(tracking PUBLIC input_dev VINSDatabase data_preprocessing tictoc feature_type database cam_odometry hardware_manager rater)
