/**
 * @file timer.cpp
 * @author Rajesh Kumar
 * @brief Implements a Timer class to measure and retrieve time in double
 * format.
 * @version 0.1
 * @date 2023-08-08
 *
 * @copyright Copyright (c) 2023
 */

#include "timer.h"

/**
 * @brief Construct a new Timer::Timer object.
 */
Timer::Timer() {
    // Ensure static variables are properly initialized (though unnecessary for
    // C++)
    time_set_ = false;
}

/**
 * @brief Destroy the Timer::Timer object.
 */
Timer::~Timer() {}

/**
 * @brief Resets the timer, setting the start time to the current time.
 */
void Timer::timeReset() {
    std::lock_guard<std::mutex> lock_guard(mutex_self_);

    start_time_ = std::chrono::high_resolution_clock::now();
    time_set_ = true;
}

/**
 * @brief Gets the current time elapsed since the last reset.
 *
 * @return double The elapsed time in seconds.
 */
double Timer::getCurTime() {
    std::lock_guard<std::mutex> lock_guard(
        mutex_self_);  // Ensure thread safety

    assert(
        time_set_ &&
        "Timer has not been set! Call timeReset() before using getCurTime().");

    // Compute elapsed time in seconds
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - start_time_)
        .count();
}

// Define static variables outside the class
bool Timer::time_set_ = false;

std::mutex Timer::mutex_self_;

std::chrono::time_point<std::chrono::high_resolution_clock> Timer::start_time_;
