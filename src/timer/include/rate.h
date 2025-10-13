/**
 * @file rate.h
 * @author Rajesh Kumar
 * @brief Defines a Rate class to control loop execution frequency.
 * @version 0.1
 * @date 2023-10-06
 *
 * @copyright Copyright (c) 2023
 */

#ifndef RATE_H_
#define RATE_H_

#include <chrono>
#include <unistd.h>

#include <thread>

/**
 * @class Rate
 * @brief A class to regulate execution frequency by controlling sleep duration.
 *
 * This class ensures that a loop runs at a specified frequency by introducing
 * appropriate sleep intervals.
 */
class Rate {
   public:
    /**
     * @brief Constructs a Rate object with a given frequency.
     * @param freq The desired frequency (Hz) at which the loop should run.
     */
    inline Rate(const int &freq)
        : freq_(freq),
          prev_time_(std::chrono::steady_clock::now()) {
            cycle_sleep_time_ = std::chrono::duration<double>(1.0 / freq_);
          }

    /**
     * @brief Destroys the Rate object.
     */
    inline ~Rate() = default;

    /**
     * @brief Sleeps the current thread to maintain the desired frequency.
     */
    void sleepThread();

    /**
     * @brief Executes a sleep interval based on elapsed time.
     */
    void sleepExe();

   private:
    /// @brief Desired execution frequency in Hz.
    int freq_;

    /// @brief Time of the previous cycle iteration.
    std::chrono::steady_clock::time_point prev_time_;

    /// @brief Computed sleep duration for each cycle.
    std::chrono::duration<double> cycle_sleep_time_;
};

#endif  // RATE_H_
