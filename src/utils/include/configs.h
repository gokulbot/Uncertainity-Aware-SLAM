//file paths 
#define MODEL_PATH "./work/models/MACVO.pth"
#define LOG_DIR "./works/logs/"

//logging macros 
#define LOG_INFO(fmt, ...) \
    Logger::getInstance().info(fmt, ##__VA_ARGS__)

#define LOG_WARN(fmt, ...) \
    Logger::getInstance().warn("[{}:{}] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
    Logger::getInstance().error("[{}:{}] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_DEBUG(fmt, ...) \
    Logger::getInstance().debug("[{}:{}] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)
