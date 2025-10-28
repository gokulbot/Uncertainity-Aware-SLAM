/**
 * @file tracking.h
 * @brief Defines the Tracking modules and its features for SLAM.
 * @author Gokul Raj Santhosh
 */

 #ifndef TRACKING_H
 #define TRACKING_H
 
 #include <atomic>
 #include <condition_variable>
 #include <map>
 #include <mutex>
 #include <queue>
 #include <thread>
 
 #include "database.h"
 #include "eigen3/Eigen/Dense"
 #include "feature_context.h"
 #include "input_device_context.h"
 #include "logger.h"
 #include "odom_frames.h"
 #include "sensor_data_preprocessing.h"
 #include "tictoc.h"
 #include "tracking_interface.h"
 #include "vins_database.h"
 #include "camera_odometry_context.h"
 #include "rate.h"
 
 /**
  * @class Tracking
  * @brief Implements tracking functionality for SLAM.
  *
  * Responsible for processing sensor data, managing tracking modules,
  * and providing the latest estimated pose.
  */
 class Tracking : public TrackingInterface {
    public:
     /**
      * @brief Constructor for Tracking.
      */
     Tracking();
 
     /**
      * @brief Destructor for Tracking.
      */
     ~Tracking();
 

     /**
      * @brief Initializes tracking with the provided setup parameters.
      *
      * @param setup The configuration for tracking initialization.
      * @return true if setup is successful, false otherwise.
      */
     bool setup(const TrackingSetupFrame& setup) override;
 
     /**
      * @brief Starts the tracking process.
      *
      * @param vins_database Unique pointer to the VINS database.
      * @param database Unique pointer to the general database.
      * @return true if tracking starts successfully, false otherwise.
      */
     bool start() override;
 
     /**
      * @brief Stops the tracking process.
      *
      * @return true if tracking stops successfully, false otherwise.
      */
     bool stop() override;
 
     /**
      * @brief Checks if tracking is currently running.
      *
      * @return true if tracking is active, false otherwise.
      */
     bool isRunning() override;
 
     /**
      * @brief Retrieves the latest tracked pose.
      *
      * @return true if pose retrieval is successful, false otherwise.
      */
     bool getTrackedPose(PoseFrame& pose) override;
 
     /**
      * @brief Retrieves the latest camera data frames.
      *
      * @param frame Reference to store the latest colour frame.
      * @param depth Reference to store the latest depth frame.
      */
     void getCameraData(ColourFrame& frame, DepthFrame& depth) override;
 
     /**
      * @brief Retrieves the latest IMU data.
      *
      * @param imu Reference to an array to store the latest IMU frames.
      */
     void getIMUData(std::array<IMUFrame, IMU_BUFFER_SIZE>& imu) override;
 
     /**
      * @brief Retrieves camera frames for further processing.
      *
      * @param frames Vector of colour frames to process.
      */
     void getCameraFrames(const std::vector<ColourFrame>& frames) override;
 
     /**
      * @brief Retrieves the latest ground truth frame.
      * @param gt_frame Reference to store the ground truth frame.
      * @note This function is used to retrieve the ground truth frame for
      * evaluation purposes. It is not used in the main SLAM pipeline.
      */
     void getGroundTruthFrame(GtFrame& gt_frame) override;
 
    private:
     /**
      * @brief Main function for tracking thread execution.
      */
     void work_(std::unique_ptr<VINSDatabase>& vins_database,
         std::unique_ptr<Database>& database, 
         std::unique_ptr<HardwareManager>& hardware_manager);
 
     /**
      * @brief Processes input dataset (called internally).
      */
     void inputDataSet_();
 
     /**
      * @brief Camera calibration parameters
      */
     CameraParams params_;
 
     /**
      * @brief Stores the status of each tracking module.
      */
     std::map<SLAMModules::TrackingModules, bool> modules_;
 
     /**
      * @brief Thread for handling odometry calculations.
      */
     std::thread odom_th_;
 
     /**
      * @brief Thread for handling input data processing.
      */
     std::thread input_th_;
 
     /**
      * @brief Atomic flag to indicate if tracking is running.
      */
     std::atomic<bool> is_running_;
 
     /**
      * @brief Unique pointer to input device context.
      */
     std::unique_ptr<InputDeviceContext> input_dev_;
 
     /**
      * @brief Unique pointer to feature context.
      */
     std::unique_ptr<FeatureContext> feature_type_;
 
     /**
      * @brief Unique pointer to initialization strategy.
      */
     std::unique_ptr<InitialisationStratergy> init_context_;
 
     /**
      * @brief Queue for IMU frame batches.
      */
     std::queue<std::vector<IMUFrame>> input_imu_queue_;
 
     /**
      * @brief Mutex for protecting access to shared pose data.
      */
     std::mutex pose_mutex_;
 
     /**
      * @brief Stores the latest estimated pose.
      */
     PoseFrame shared_pose_;
 
     /**
      * @brief Stores the latest received color frame.
      */
     ColourFrame color_frame_;
 
     /**
      * @brief Stores the latest received stereo frame.
      */
     StereoFrame stereo_frame_;
 
     /**
      * @brief Stores the latest received depth frame.
      */
     DepthFrame depth_frame_;
 
     /**
      * @brief Configuration setup for input devices.
      */
     InputDevSetup input_dev_setup_;
 
     /**
      * @brief Configuration for feature tracking.
      */
     FeatureConfig feature_config_;
 
     /**
      * @brief Pointer to the cam odometry module.
      */
     std::unique_ptr<CameraOdometryContext> cam_odom_;
 
     /**
      * @brief Mutex and condition variable to synchronize input frame
      * processing.
      */
     std::mutex input_frame_mutex_;
     std::condition_variable input_frame_cv_;
 
     /**
      * @brief Queues to store incoming frames with buffer size limits.
      */
     std::queue<StereoFrame> input_frame_queue_;
     std::queue<DepthFrame> input_depth_queue_;
     static constexpr size_t INPUT_FRAME_BUFFER_SIZE = 10;
 
     /**
      * @brief previous pose data
      *
      */
     PoseFrame prev_pose_;
 
     /**
      * @brief previous pose data
      *
      */
     PoseFrame current_pose_;
 
     /**
      * @brief intial pose data
      *
      */
     PoseFrame initial_pose_;
 
     /**
      * @brief Current keyframe created
      *
      */
     KF_ID curr_kf_;
 
     /**
      * @brief Previous keyframe created
      *
      */
     KF_ID prev_kf_;
 
     std::queue<ColourFrame> input_color_queue_;
 
     /**
      * @brief system intiaization flag
      */
 
     bool system_intailized_ = false;
 
     /**
      * @brief Latest feature extracted from the odometry module.
      */
     Feature latest_feature_;
 
     /**
      * @brief Latest depth frame extracted from the odometry module.
      */
     DepthFrame latest_depth_;
 
     /**
      * @brief Time for the current frame being processed.
      */
     double curr_image_time_ = 0.0;
 };
 
 #endif  // TRACKING_H
 