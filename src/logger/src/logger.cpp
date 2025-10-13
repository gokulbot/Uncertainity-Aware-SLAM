/**
 * @file logger.cpp
 * @author Gokul Raj Santhosh
 * @brief Implementation of the Logger class using spdlog.
 * @details Provides thread-safe logging methods for info, warn, error, and
 * debug levels.
 */
#include "logger.h"

/**
 * @brief Constructor for Logger class.
 * Initializes the logger with a color sink and sets the log pattern.
 * @details The logger is set to log messages at the debug level.
 * This constructor is private to enforce the singleton pattern.
 */
Logger::Logger() {
    std::string log_dir = LOG_DIR;
    std::filesystem::create_directories(
        log_dir);  // Ensure the directory exists

    // Create a file sink
    std::string slam_log_file = log_dir + "slam.log";
    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        slam_log_file,    // base log file name
        1024 * 1024 * 5,  // 5 MB size limit per file
        3                 // keep up to 3 rotated files
    );

    // Optionally add a colored console sink as well
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    // Create a logger with both sinks
    logger_ = std::make_shared<spdlog::logger>(
        "multi_sink", spdlog::sinks_init_list{console_sink, rotating_sink});

    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    logger_->set_level(spdlog::level::debug);
}

/**
 * @brief Sets the logging level for the logger.
 * @param level The logging level to set.
 * @details This method uses a scoped lock to ensure thread safety while setting
 * the log level.
 */
void Logger::setLevel(spdlog::level::level_enum level) {
    std::lock_guard<std::mutex> lock(mutex_);
    logger_->set_level(level);
}

void Logger::setLevelFromString(const std::string& levelStr) {
    auto level = spdlog::level::from_str(levelStr);
    setLevel(level);
}

/**
 * @brief Returns the singleton instance of Logger.
 * @details Uses a static local variable to ensure that only one instance of
 * Logger is created. This method is thread-safe and guarantees that the logger
 * is initialized only once.
 * @return Reference to the Logger instance.
 */
Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}