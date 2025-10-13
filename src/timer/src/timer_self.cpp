/**
 * @file timer_self.cpp
 * @author Rajesh Kumar
 * @brief Implements a TimerSelf class for measuring elapsed time.
 * @version 0.1
 * @date 2023-11-01
 *
 * @copyright Copyright (c) 2023
 */

#include "timer_self.h"

/**
 * @brief Resets the timer to start measuring from the current time.
 */
void TimerSelf::timeReset() {
    start_time_ = std::chrono::high_resolution_clock::now();
    time_set_ = true;
}

/**
 * @brief Retrieves the elapsed time since the last reset.
 *
 * @return double Elapsed time in seconds.
 */
double TimerSelf::getCurTime() {
    assert(
        time_set_ &&
        "Timer has not been set! Call timeReset() before using getCurTime().");

    // Compute the time difference in seconds
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - start_time_)
        .count();
}
