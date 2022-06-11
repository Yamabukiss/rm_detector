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

  void receiveFromCam(const sensor_msgs::ImageConstPtr& image);

  void staticResize(cv::Mat& img, const float& scale);

  float* blobFromImage(cv::Mat& img);

  void generateGridsAndStride(const int target_w, const int target_h);

  void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold,
                              std::vector<Object>& proposals, const int& NUM_CLASSES);

  inline float intersectionArea(const Object& a, const Object& b);

  void qsortDescentInplace(std::vector<Object>& faceobjects, int left, int right);

  void qsortDescentInplace(std::vector<Object>& proposals);

  void nmsSortedBboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);

  void drawObjects(const cv::Mat& bgr);

  void mainFuc(cv_bridge::CvImagePtr& image_ptr);

  void initalizeInferOfCar();
  void initalizeInferOfArmor();

  void dynamicCallback(rm_detector::dynamicConfig& config);

  void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, const int output_size,
                   const cv::Size& input_shape);

  void publishDataForRed(const Object& object);
  void publishDataForBlue(const Object& object);
  //  void publishUndetectableNum(std::vector<int> detectable_vec, std::vector<int> color_num_vec,
  //                              std::vector<Object> objects, int img_w, int img_h);
  void getRoiImg(const std::vector<Object>& object, std::vector<cv::Mat>& roi_vec);
  void detectArmor(std::vector<cv::Mat>& roi_vec);

  ros::NodeHandle nh_;
  cv_bridge::CvImagePtr cv_image_{};
  std::vector<GridAndStride> grid_strides_;
  float nms_thresh_;
  float nms_thresh2_;
  float bbox_conf_thresh_;
  float bbox_conf_thresh2_;
  std_msgs::Float32MultiArray roi_data_;
  std::vector<cv::Point2f> roi_point_vec_;
  cv::Point2f roi_data_point_r_;
  cv::Point2f roi_data_point_l_;
  cv::Mat_<float> discoeffs_;
  cv::Mat_<float> camera_matrix_;
  std::vector<float> discoeffs_vec_;
  std::vector<float> camera_matrix_vec_;
  Object armor_object_;
  std::vector<Object> car_objects_;
  std::vector<Object> prev_objects_;
  std::vector<int> blue_lable_vec;
  std::vector<int> red_lable_vec;
  std::string car_model_path_;
  std::string armor_model_path_;
  std::string model_path_;
  float scale_;
  float scale2_;
  bool turn_on_image_;
  dynamic_reconfigure::Server<rm_detector::dynamicConfig>* server_;
  dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback_;
  std::string nodelet_name_;
  std::string camera_pub_name_;
  std::string roi_data1_name_;
  std::string roi_data2_name_;
  std::string roi_data3_name_;
  std::string roi_data4_name_;
  std::string roi_data5_name_;
  bool left_camera_;
  bool target_is_red_;
  bool target_is_blue_;
  cv::Mat roi_picture_;
  std::vector<cv::Mat> roi_picture_vec_;
  std::vector<cv::Mat> roi_picture_split_vec_;
  std::vector<Object> filter_objects_;
  char* trt_model_stream_{};
  char* trt_model_stream2_{};
  nvinfer1::IRuntime* runtime_{};
  nvinfer1::IRuntime* runtime2_{};
  nvinfer1::ICudaEngine* engine_{};
  nvinfer1::ICudaEngine* engine2_{};
  nvinfer1::IExecutionContext* context_{};
  nvinfer1::IExecutionContext* context2_{};
  float* prob_{};
  float* prob2_{};
  int output_size_;
  int output_size2_;
  std::vector<Object> armor_object_vec_;

private:
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
}  // namespace rm_detector