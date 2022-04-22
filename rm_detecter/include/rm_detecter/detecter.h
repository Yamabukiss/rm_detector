//
// Created by yamabuki on 2022/4/18.
//
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <memory>
#include <string>
#include <vector>
#include <inference_engine.hpp>
#include "std_msgs/Int32MultiArray.h"
#include "dynamic_reconfigure/server.h"
#include "rm_detecter/dynamicConfig.h"

struct Object
{
    cv::Rect_<float> rect;
    cv::Point2f lu;
    cv::Point2f ld;
    cv::Point2f ru;
    cv::Point2f rd;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

namespace rm_detecter
{
class Detecter
{
public:
    Detecter();
  virtual ~Detecter();

  void initialize( const ros::NodeHandle& nh) ;
  void receiveFromCam(const sensor_msgs::ImageConstPtr& image) ;
     void sendMsg(boost::shared_ptr<sensor_msgs::Image> msg);
    cv::Mat staticResize(cv::Mat& img);
    void blobFromImage(cv::Mat& img, InferenceEngine::MemoryBlob::Ptr &mblob);
    static void generateGridsAndStride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
    static void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects);
    static inline float intersectionArea(const Object& a, const Object& b);
    static void qsortDescentInplace(std::vector<Object>& faceobjects, int left, int right);
    static void qsortDescentInplace(std::vector<Object>& objects);
    static void nmsSortedBboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
    static void decodeOutputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
    sensor_msgs::ImagePtr drawObjects(const cv::Mat& bgr, const std::vector<Object>& objects);
      sensor_msgs::ImagePtr mainFuc( cv_bridge::CvImagePtr& image_ptr,std::vector<Object> objects);
      bool getCam(cv_bridge::CvImagePtr& cv_image, sensor_msgs::CameraInfoConstPtr& cam_info) ;
  inline void sendROI(std_msgs::Int32MultiArray roi_data);
    void  initalize_infer();
  static cv_bridge::CvImagePtr cv_image_;
  static sensor_msgs::CameraInfoConstPtr info_;
  static InferenceEngine::InferRequest infer_request_;
  static InferenceEngine::MemoryBlob::Ptr mblob_;
  static const float* net_pred_;

private:
  ros::NodeHandle nh_;
  ros::Publisher camera_pub_;
  ros::Subscriber camera_sub_;
//  std::shared_ptr<image_transport::ImageTransport> it_;
//  image_transport::CameraSubscriber camera_sub_;
  ros::Publisher roi_data_pub_;
//  image_transport::Publisher camera_pub_;


};
}  // namespace rm_receiver