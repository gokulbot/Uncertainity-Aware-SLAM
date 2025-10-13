/**
 * @file timer_self.h
 * @author Rajesh Kumar
 * @brief Defines a TimerSelf class for measuring elapsed time.
 * @version 0.1
 * @date 2023-11-01
 *
 * @copyright Copyright (c) 2023
 */

#ifndef TIMER_SELF_H_
#define TIMER_SELF_H_

#include <cassert>
#include <chrono>
#include <ctime>

/**
 * @class TimerSelf
 * @brief A simple high-resolution timer class.
 *
 * This class provides functionality to measure uninterrupted
 * time using a high-resolution clock.
 */
class TimerSelf {
   public:
    /**
     * @brief Construct a new TimerSelf object.
     */
    TimerSelf() = default;

    /**
     * @brief Destroy the TimerSelf object.
     */
    ~TimerSelf() = default;

    /**
     * @brief Resets the timer to start measuring from the current time.
     */
    void timeReset();

    /**
     * @brief Gets the elapsed time in seconds since the last reset.
     * @return double Elapsed time in seconds.
     */
    double getCurTime();

   private:
    /// @brief Indicates whether the timer has been initialized.
    bool time_set_;

    /// @brief Stores the starting time of the timer.
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

#endif  // TIMER_SELF_H_
