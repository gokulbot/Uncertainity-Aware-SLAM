#pragma once
#include <string>
#include "configs.h"
#include <opencv2/opencv.hpp>

typedef struct ModelSetup {

    std::string model_path;

    void setDefault(){
        model_path = MODEL_PATH;
    }
}ModelSetup;

/**
 * @brief Represents a stereo image frame.
 * @struct StereoFrame
 */
typedef struct StereoFrame {
    /**
     * @brief Grayscale image from the left camera.
     */
    cv::Mat gray_frame;

    /**
     * @brief Grayscale image from the right camera.
     */
    cv::Mat gray_frame_right;

    /**
     * @brief Timestamp of the stereo frame.
     */
    double time = 0;

    /**
     * @brief Display combined left and right grayscale images side-by-side.
     * @param name Name of the display window.
     */
    void vis(std::string name) const {
        int totalWidth = gray_frame.cols + gray_frame.cols;
        int height = gray_frame.rows;
        cv::Mat combined(height, totalWidth, gray_frame.type());

        gray_frame.copyTo(
            combined(cv::Rect(0, 0, gray_frame.cols, gray_frame.rows)));
        gray_frame_right.copyTo(combined(cv::Rect(
            gray_frame.cols, 0, gray_frame_right.cols, gray_frame_right.rows)));

        cv::imshow(name, combined);
        cv::waitKey(1);
    }

    /**
     * @brief Compare raw and processed grayscale frames visually.
     * @param name Window name.
     * @param gray_frame_raw Raw left frame.
     * @param gray_frame_right_raw Raw right frame.
     */
    void visComparison(const std::string& name, const cv::Mat& gray_frame_raw,
                       const cv::Mat& gray_frame_right_raw) {
        cv::Mat left_combined;
        cv::hconcat(gray_frame_raw, gray_frame, left_combined);

        cv::Mat right_combined;
        cv::hconcat(gray_frame_right_raw, gray_frame_right, right_combined);

        cv::Mat final_display;
        cv::vconcat(left_combined, right_combined, final_display);

        cv::imshow(name, final_display);
        cv::waitKey(1);
    }

    /**
     * @brief Clone the stereo frame (deep copy).
     * @return Copied StereoFrame.
     */
    StereoFrame clone() const {
        StereoFrame copy;
        copy.gray_frame = gray_frame.clone();
        copy.gray_frame_right = gray_frame_right.clone();
        copy.time = time;
        return copy;
    }

    /**
     * @brief Destroy the visualization window.
     * @param name Window name.
     */
    void destroy(std::string name) { cv::destroyWindow(name); }

} StereoFrame;

/**
 * @brief Input structure for MACVO model.
 * @struct MACVOInput
 * @details Contains stereo frames and flags for processing.
 * @var input_frame StereoFrame containing left and right images.
 * @var is_keyframe Boolean flag indicating if the frame is a keyframe.
 * @note Default value for is_keyframe is false.
 * @see StereoFrame
 * 
 */
struct MACVOInput{

    StereoFrame input_frame;
    bool is_keyframe = false;

};

/**
 * @brief Output structure for MACVO model.
 * @struct MACVOOutput
 * @details Contains pose, depth map, and depth variance.
 * @var pose 4x4 transformation matrix (cv::Mat).
 * @var depth Depth map (cv::Mat).
 * @var depth_variance Variance of the depth map (cv::Mat).
 * @note All matrices are of type CV_32F.
 * @see cv::Mat
 */

struct MACVOOutput{

    cv::Mat pose;
    cv::Mat depth;
    cv::Mat depth_variance;

};