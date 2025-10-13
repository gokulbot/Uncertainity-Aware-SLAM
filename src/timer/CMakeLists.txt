cmake_minimum_required(VERSION 3.10)
project(timer)

#################### TIMER H #########################
add_library(timer SHARED src/timer.cpp)

target_include_directories(timer PUBLIC include)


#################### RATER H ##########################
add_library(rater SHARED src/rate.cpp)

target_include_directories(rater PUBLIC include)
target_link_libraries(rater PUBLIC timer)


############### TIMER SELF #######################

add_library(timer_self SHARED src/timer_self.cpp)

target_include_directories(timer_self PUBLIC include)