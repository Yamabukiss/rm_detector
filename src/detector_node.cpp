//
// Created by dynamicx on 22-6-8.
//

#include "rm_detector/detector.h"

int main(int argc,char** argv)
{
    ros::init(argc,argv,"armor_detect_nodelet");

    rm_detector::Detector detector;
    detector.nh_ = ros::NodeHandle("~");
    detector.onInit();

    ros::spin();
    return 0;
}