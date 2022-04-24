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
#include <inference_engine.hpp>
#include "std_msgs/Int16MultiArray.h"
#include "dynamic_reconfigure/server.h"
#include "rm_detector/dynamicConfig.h"
#include "sensor_msgs/CameraInfo.h"
#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>

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
  cv::Mat staticResize(cv::Mat& img);
  void blobFromImage(cv::Mat& img, InferenceEngine::MemoryBlob::Ptr& mblob);
  void generateGridsAndStride(const int target_w, const int target_h);
  void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold,
                              std::vector<Object>& objects);
  inline float intersectionArea(const Object& a, const Object& b);
  void qsortDescentInplace(std::vector<Object>& faceobjects, int left, int right);
  void qsortDescentInplace(std::vector<Object>& objects);
  void nmsSortedBboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
  void decodeOutputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
  void drawObjects(const cv::Mat& bgr, const std::vector<Object>& objects);
  void mainFuc(cv_bridge::CvImagePtr& image_ptr, std::vector<Object> objects);
  void initalize_infer();
  void dynamic_callback(rm_detector::dynamicConfig& config);
  cv_bridge::CvImagePtr cv_image_;
  InferenceEngine::InferRequest infer_request_;
  InferenceEngine::MemoryBlob::Ptr mblob_;
  const float* net_pred_;
  std::vector<GridAndStride> grid_strides_;
  float nms_thresh_;
  float bbox_conf_thresh_;
  std_msgs::Int16MultiArray roi_data_;
  std::vector<Object> objects_;
  std::string model_path_;
  bool turn_on_image_;
  dynamic_reconfigure::Server<rm_detector::dynamicConfig> server_;
  dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback_;

private:
  ros::NodeHandle nh_;
  ros::Publisher camera_pub_;
  ros::Subscriber camera_sub_;
  std::vector<ros::Publisher> roi_data_pub_vec;
  ros::Publisher roi_data_pub1_;
  ros::Publisher roi_data_pub2_;
  ros::Publisher roi_data_pub3_;
};
}  // namespace rm_detector