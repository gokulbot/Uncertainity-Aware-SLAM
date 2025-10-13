/**
 * @file rate.cpp
 * @author Rajesh Kumar
 * @brief Implements a Rate class to control loop execution frequency.
 * @version 0.1
 * @date 2023-10-06
 *
 * @copyright Copyright (c) 2023
 */

#include "rate.h"
#include <iostream>

/**
 * @brief Sleeps the current thread to maintain the desired execution rate.
 *
 * This function ensures that the loop runs at a consistent frequency
 * by computing the elapsed time and sleeping for the remaining cycle duration.
 */
void Rate::sleepThread() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - prev_time_;
    std::chrono::duration<double> sleep_time = cycle_sleep_time_ - elapsed_time;

    if (sleep_time.count() > 0) {
        std::this_thread::sleep_for(sleep_time);
    }

    prev_time_ = std::chrono::steady_clock::now();
}

/**
 * @brief Sleeps the entire executable to maintain the desired execution rate.
 *
 * Unlike `sleepThread()`, which only pauses the calling thread, this function
 * suspends the entire program execution for the required duration.
 */
void Rate::sleepExe() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - prev_time_;
    std::chrono::duration<double> sleep_time = cycle_sleep_time_ - elapsed_time;

    if (sleep_time.count() > 0) {
        sleep(static_cast<unsigned int>(
            sleep_time.count()));  // sleep() expects an unsigned int in seconds
    }

    prev_time_ = std::chrono::steady_clock::now();
}
