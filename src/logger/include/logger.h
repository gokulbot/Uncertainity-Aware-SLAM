/**
 * @file logger.h
 * @author Gokul Raj Santhosh
 * @brief Singleton logger class using spdlog for logging messages.
 * @details Provides thread-safe logging methods for info, warn, error, and
 * debug levels. Uses variadic templates for formatted logging.
 * @version 1.0
 */
#ifndef LOGGER_H
#define LOGGER_H

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <mutex>

#include "configs.h"

/**
 * @class Logger
 * @brief Singleton wrapper around spdlog for centralized logging.
 */
class Logger {
   public:
    static Logger& getInstance();

    // Variadic template logging methods (fmt-style)
    /**
     * @brief Logs an info message with formatted string.
     * @param fmt Format string for the log message.
     * @param args Arguments to format the string.
     * @details Uses std::scoped_lock to ensure thread safety.
     *          The logger must be initialized before calling this method.
     */
    template <typename... Args>
    void info(const std::string& fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        logger_->info(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Logs a warning message with formatted string.
     * @param fmt Format string for the log message.
     * @param args Arguments to format the string.
     * @details Uses std::scoped_lock to ensure thread safety.
     *          The logger must be initialized before calling this method.
     */
    template <typename... Args>
    void warn(const std::string& fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        logger_->warn(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Logs an error message with formatted string.
     * @param fmt Format string for the log message.
     * @param args Arguments to format the string.
     * @details Uses std::scoped_lock to ensure thread safety.
     *          The logger must be initialized before calling this method.
     */
    template <typename... Args>
    void error(const std::string& fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        logger_->error(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Logs a debug message with formatted string.
     * @param fmt Format string for the log message.
     * @param args Arguments to format the string.
     * @details Uses std::scoped_lock to ensure thread safety.
     *          The logger must be initialized before calling this method.
     */
    template <typename... Args>
    void debug(const std::string& fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        logger_->debug(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Sets the logging level from a string.
     * @param levelStr String representation of the logging level (e.g., "info",
     * "warn", "error", "debug").
     * @details Converts the string to the corresponding spdlog level and sets
     * it. If the string does not match any known level, it defaults to debug
     * level.
     * @note This method is thread-safe and can be called at any time to change
     * the logging level.
     */
    void setLevelFromString(const std::string& levelStr);

    /**
     * @brief Sets the logging level.
     * @param level The logging level to set (spdlog::level::level_enum).
     * @details This method uses a scoped lock to ensure thread safety while
     * setting the log level.
     */
    void setLevel(spdlog::level::level_enum level);

   private:
    // private constructor to prevent instantiation
    Logger();  // defined in .cpp

    /**
     * @brief Pointer to the spdlog logger instance.
     * @details This logger is initialized in the constructor and used for
     * logging messages.
     */
    std::shared_ptr<spdlog::logger> logger_;

    /**
     * @brief Mutex for thread-safe access to the logger.
     * @details Ensures that only one thread can log messages at a time,
     */
    std::mutex mutex_;
};

#endif  // LOGGER_H
