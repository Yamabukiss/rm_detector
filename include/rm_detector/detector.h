//
// Created by yamabuki on 2022/4/18.
//
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Int8.h"
#include "dynamic_reconfigure/server.h"
#include "rm_detector/dynamicConfig.h"
#include "sensor_msgs/CameraInfo.h"
#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

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
class Detector : public nodelet::Nodelet
{
public:
  Detector();

  virtual ~Detector();

  void onInit() override;

  void receiveFromLeftCam(const sensor_msgs::ImageConstPtr& image);
  void receiveFromRightCam(const sensor_msgs::ImageConstPtr& image);

  void staticResize(cv::Mat& img, const float& scale);

  float* blobFromImage(cv::Mat& img);

  void generateGridsAndStrideLeft(const int target_w, const int target_h);
  void generateGridsAndStrideRight(const int target_w, const int target_h);

  void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold,
                              std::vector<Object>& proposals, const int& NUM_CLASSES);

  inline float intersectionArea(const Object& a, const Object& b);

  void qsortDescentInplace(std::vector<Object>& faceobjects, int left, int right);

  void qsortDescentInplace(std::vector<Object>& proposals);

  void nmsSortedBboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);

  void drawObjectsLeft(const cv::Mat& bgr);
  void drawObjectsRight(const cv::Mat& bgr);

  void mainFucLeft(cv_bridge::CvImagePtr& image_ptr);
  void mainFucRight(cv_bridge::CvImagePtr& image_ptr);

  void initalizeInferOfCarLeft();
  void initalizeInferOfArmorLeft();
  void initalizeInferOfCarRight();
  void initalizeInferOfArmorRight();

  void dynamicCallback(rm_detector::dynamicConfig& config);

  void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, const int output_size,
                   const cv::Size& input_shape);

  void publishDataForRedLeft(const Object& object);
  void publishDataForBlueLeft(const Object& object);

  void publishDataForRedRight(const Object& object);
  void publishDataForBlueRight(const Object& object);

  //  void publishUndetectableNum(std::vector<int> detectable_vec, std::vector<int> color_num_vec,
  //                              std::vector<Object> objects, int img_w, int img_h);
  void getRoiImgLeft(const std::vector<Object>& object, std::vector<cv::Mat>& roi_vec);
  void getRoiImgRight(const std::vector<Object>& object, std::vector<cv::Mat>& roi_vec);

  void detectArmorLeft(std::vector<cv::Mat>& roi_vec);
  void detectArmorRight(std::vector<cv::Mat>& roi_vec);

  ros::NodeHandle nh_;
  cv_bridge::CvImagePtr cv_image_left_{};
  cv_bridge::CvImagePtr cv_image_right_{};
  std::vector<GridAndStride> grid_strides_left_;
  std::vector<GridAndStride> grid_strides_right_;
  float nms_thresh_left_;
  float nms_thresh2_left_;
  float nms_thresh_right_;
  float nms_thresh2_right_;
  float bbox_conf_thresh_left_;
  float bbox_conf_thresh2_left_;
  float bbox_conf_thresh_right_;
  float bbox_conf_thresh2_right_;
  std_msgs::Float32MultiArray roi_data_left_;
  std_msgs::Float32MultiArray roi_data_right_;
  std::vector<cv::Point2f> roi_point_vec_left_;
  std::vector<cv::Point2f> roi_point_vec_right_;
  cv::Point2f roi_data_point_r_left_;
  cv::Point2f roi_data_point_l_left_;
  cv::Point2f roi_data_point_r_right_;
  cv::Point2f roi_data_point_l_right_;
  cv::Mat_<float> discoeffs_;
  cv::Mat_<float> camera_matrix_;
  std::vector<float> discoeffs_vec_;
  std::vector<float> camera_matrix_vec_;
  Object armor_object_;
  std::vector<Object> car_objects_left_;
  std::vector<Object> prev_objects_left_;
  std::vector<Object> car_objects_right_;
  std::vector<Object> prev_objects_right_;
  std::vector<int> blue_lable_vec_left_;
  std::vector<int> red_lable_vec_left_;
  std::vector<int> blue_lable_vec_right_;
  std::vector<int> red_lable_vec_right_;
  std::string car_model_path_left_;
  std::string armor_model_path_left_;
  std::string car_model_path_right_;
  std::string armor_model_path_right_;
  std::string model_path_;

  float scale_left_;
  float scale2_left;
  float scale_right_;
  float scale2_right_;

  bool turn_on_image_left_;
  bool turn_on_image_right_;
  dynamic_reconfigure::Server<rm_detector::dynamicConfig>* server_;
  dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback_;
  std::string nodelet_name_;
  std::string camera_pub_name_;
  std::string roi_data1_name_;
  std::string roi_data2_name_;
  std::string roi_data3_name_;
  std::string roi_data4_name_;
  std::string roi_data5_name_;
  std::string roi_data6_name_;
  std::string roi_data7_name_;
  std::string roi_data8_name_;
  std::string roi_data9_name_;
  std::string roi_data10_name_;
  bool left_camera_;
  bool target_is_red_left_;
  bool target_is_blue_left_;
  bool target_is_red_right_;
  bool target_is_blue_right_;
  cv::Mat roi_picture_;
  std::vector<cv::Mat> roi_picture_vec_;
  std::vector<cv::Mat> roi_picture_split_vec_;
  std::vector<Object> filter_objects_;
  char* trt_model_stream_left_{};
  char* trt_model_stream2_left_{};
  char* trt_model_stream_right_{};
  char* trt_model_stream2_right_{};
  nvinfer1::IRuntime* runtime_left_{};
  nvinfer1::IRuntime* runtime2_left_{};
  nvinfer1::IRuntime* runtime_right_{};
  nvinfer1::IRuntime* runtime2_right_{};
  nvinfer1::ICudaEngine* engine_left_{};
  nvinfer1::ICudaEngine* engine2_left_{};
  nvinfer1::ICudaEngine* engine_right_{};
  nvinfer1::ICudaEngine* engine2_right_{};
  nvinfer1::IExecutionContext* context_left_{};
  nvinfer1::IExecutionContext* context2_left{};
  nvinfer1::IExecutionContext* context_right_{};
  nvinfer1::IExecutionContext* context2_right_{};
  float* prob_left_{};
  float* prob2_left_{};
  float* prob_right_{};
  float* prob2_right_{};
  int output_size_left_;
  int output_size2_left_;
  int output_size_right_;
  int output_size2_right_;
  std::vector<Object> armor_object_vec_left_;
  std::vector<Object> armor_object_vec_Right_;

private:
  ros::Publisher camera_pub_;
  ros::Publisher camera_pub2_;
  ros::Subscriber camera_sub_left_;
  ros::Subscriber camera_sub_right_;
  std::vector<ros::Publisher> roi_data_pub_vec_l;
  std::vector<ros::Publisher> roi_data_pub_vec_r;
  ros::Publisher roi_data_pub1_;
  ros::Publisher roi_data_pub2_;
  ros::Publisher roi_data_pub3_;
  ros::Publisher roi_data_pub4_;
  ros::Publisher roi_data_pub5_;
  ros::Publisher roi_data_pub6_;
  ros::Publisher roi_data_pub7_;
  ros::Publisher roi_data_pub8_;
  ros::Publisher roi_data_pub9_;
  ros::Publisher roi_data_pub10_;
};
}  // namespace rm_detector