//
// Created by yamabuki on 2022/4/18.
//
#include <rm_detector/detector.h>

extern float g_nms_thresh;
extern float g_bbox_conf_thresh;
void dynamic_callback(rm_detector::dynamicConfig &config)
{
    g_nms_thresh = config.g_nms_thresh;
    g_bbox_conf_thresh = config.g_bbox_conf_thresh;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rm_detector");
    ros::NodeHandle nh("~");

    dynamic_reconfigure::Server<rm_detector::dynamicConfig> server;
    dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback;
    callback = boost::bind(&dynamic_callback, _1);
    server.setCallback(callback);

    rm_detector::Detector detector;
    detector.initialize(nh);
    detector.initalize_infer();
    cv_bridge::CvImagePtr cv_image;
    sensor_msgs::CameraInfoConstPtr cam_info;
    std::vector<Object> objects;

    while (ros::ok())
  {
      detector.getCam(cv_image, cam_info);

    if (cv_image.get() != nullptr)
    {
        detector.sendMsg(detector.mainFuc(cv_image,objects));
        objects.clear();
    }

    ros::spinOnce();
  }

  return 0;
}
