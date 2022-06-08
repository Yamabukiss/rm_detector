#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <nodelet/nodelet.h>
#include <std_msgs/Float32MultiArray.h>
#include <dynamic_reconfigure/server.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "ros/ros.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

namespace rm_detector
{
    class Detector: public nodelet::Nodelet
    {
    public:
        Detector();
        virtual ~Detector();
        void onInit() override;
        void receiveFromCam(const sensor_msgs::ImageConstPtr& image);
        cv_bridge::CvImagePtr cv_image_;
        const float* net_pred_;
        std::vector<GridAndStride> grid_strides_;
        float nms_thresh_;
        float bbox_conf_thresh_;
        std_msgs::Float32MultiArray roi_data_;
        std::vector<cv::Point2f> roi_point_vec_;
        cv::Point2f roi_data_point_r_;
        cv::Point2f roi_data_point_l_;
        cv::Mat_<float> discoeffs_;
        cv::Mat_<float> camera_matrix_;
        std::vector<float> discoeffs_vec_;
        std::vector<float> camera_matrix_vec_;
        std::vector<Object> objects_;
        std::string model_path_;
        int origin_img_w_;
        int origin_img_h_;
        float scale_;
        bool turn_on_image_;
        dynamic_reconfigure::Server<rm_detector::dynamicConfig> server_;
        dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback_;
        std::string nodelet_name_;
        std::string camera_pub_name_;
        std::string roi_data1_name_;
        std::string roi_data2_name_;
        std::string roi_data3_name_;
        std::string roi_data4_name_;
        std::string roi_data5_name_;
        bool target_is_red_;
        bool target_is_blue_;
        cv::Mat roi_picture_;
        std::vector<cv::Mat> roi_picture_vec_;
        std::vector<cv::Mat> roi_picture_split_vec_;
        float ratio_of_pixels_;
        int  counter_of_pixel_;
        int pixels_thresh_;
        std::vector<Object> filter_objects_;
        int binary_threshold_;
        float aspect_ratio_;

    private:
        ros::NodeHandle nh_;
        ros::Publisher camera_pub_;
        ros::Publisher camera_pub2_;
        ros::Subscriber camera_sub_;
        std::vector<ros::Publisher> roi_data_pub_vec;
        ros::Publisher roi_data_pub1_;
        ros::Publisher roi_data_pub2_;
        ros::Publisher roi_data_pub3_;
        ros::Publisher roi_data_pub4_;
        ros::Publisher roi_data_pub5_;
    };
}