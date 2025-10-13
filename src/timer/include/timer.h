/**
 * @file timer.h
 * @author Rajesh Kumar
 * @brief Defines a Timer class to measure and retrieve time in double format.
 * @version 0.1
 * @date 2023-08-08
 *
 * @copyright Copyright (c) 2023
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <cassert>
#include <chrono>
#include <ctime>
#include <mutex>

/**
 * @class Timer
 * @brief A utility class for high-precision timing.
 *
 * This class provides functionality to reset and retrieve the current time
 * as a double value using a high-resolution clock.
 */
class Timer {
   public:
    /**
     * @brief Constructs a Timer object.
     */
    Timer();

    /**
     * @brief Destroys the Timer object.
     */
    ~Timer();

    /**
     * @brief Resets the timer to start measuring from the current time.
     */
    static void timeReset();

    /**
     * @brief Retrieves the current time in seconds since the last reset.
     * @return double The elapsed time in seconds.
     */
    static double getCurTime();

   private:
    /// @brief Indicates whether the timer has been initialized.
    static bool time_set_;

    /// @brief Stores the starting time of the timer.
    static std::chrono::time_point<std::chrono::high_resolution_clock>
        start_time_;

    /// @brief Mutex to ensure thread safety while modifying the timer.
    static std::mutex mutex_self_;
};

#endif  // TIMER_H_
