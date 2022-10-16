
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>
#define INPUT_W 320
#define INPUT_H 320
#define NUM_CLASSES 2
int num=1;
namespace rm_detector
{
Detector::Detector()
{
  mblob_ = nullptr;
}

void Detector::onInit()
{
  ros::NodeHandle nh = getMTPrivateNodeHandle();
  nh.getParam("g_model_path", model_path_);
  nh.getParam("nodelet_name", nodelet_name_);
  nh.getParam("camera_pub_name", camera_pub_name_);
    initalizeInfer();
    callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
    server_.setCallback(callback_);

    nh_ = ros::NodeHandle(nh, nodelet_name_);
  camera_sub_ = nh_.subscribe("/galaxy_camera/image_raw", 1, &Detector::receiveFromCam, this);
  camera_pub_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_, 1);
  camera_pub2_ = nh_.advertise<sensor_msgs::Image>("sub_publisher", 1);

  generateGridsAndStride(INPUT_W, INPUT_H);  // the wide height strides need to be changed depending on demand
}

void Detector::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
  cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
  std::string path="/home/yamabuki/mineral/mineral3_"+std::to_string(num) + ".jpg";
  if (save_on_)
    {
        cv::imwrite(path,cv_image_->image);
        std::cout<<path<<std::endl;
        num++;
    }
    cv::waitKey(100);
//  cv::imwrite()
//  mainFuc(cv_image_);
//  objects_.clear();
//  roi_picture_vec_.clear();
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
  nms_thresh_ = config.g_nms_thresh;
  bbox_conf_thresh_ = config.g_bbox_conf_thresh;
  turn_on_image_ = config.g_turn_on_image;
  save_on_=config.save_on;
  ROS_INFO("Settings have been seted");
}

void Detector::staticResize(cv::Mat &img)
{
  int unpad_w = scale_ * img.cols;
  int unpad_h = scale_ * img.rows;
  int resize_unpad = std::max(unpad_h, unpad_w);
  origin_img_w_ = img.cols;
  origin_img_h_ = img.rows;
    cv::resize(img, img, cv::Size(resize_unpad, resize_unpad));
    cv::copyMakeBorder(img, img, abs(img.rows - img.cols) / 2, abs(img.rows - img.cols) / 2, 0, 0, cv::BORDER_CONSTANT,
                     cv::Scalar(144, 144, 144));
  img/=255;
}

void Detector::blobFromImage(cv::Mat& img)
{
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  if (!mblob_)
  {
    THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                       << "but by fact we were not able to cast inputBlob to MemoryBlob";
  }
  // locked memory holder should be alive all time while access to its buffer happens
  auto mblob_holder = mblob_->wmap();

  float* blob_data = mblob_holder.as<float*>();

  for (size_t c = 0; c < channels; c++)
  {
    for (size_t h = 0; h < img_h; h++)
    {
      for (size_t w = 0; w < img_w; w++)
      {
        blob_data[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}

// if the 3 parameters is fixed the strides can be fixed too
void Detector::generateGridsAndStride(const int target_w, const int target_h)
{
  std::vector<int> strides = { 8, 16, 32 };
  std::vector<GridAndStride> grid_strides;
  for (auto stride : strides)
  {
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++)
    {
      for (int g0 = 0; g0 < num_grid_w; g0++)
      {
        grid_strides.push_back((GridAndStride){ g0, g1, stride });
      }
    }
  }

  grid_strides_ = grid_strides;
}

void Detector::generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr,
                                      float prob_threshold, std::vector<Object>& proposals)
{
  const int num_anchors = grid_strides.size();
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
  {

    const int basic_pos = anchor_idx * (NUM_CLASSES + 9);

      float x1 = (feat_ptr[basic_pos + 0]  )  ;
      float y1 = (feat_ptr[basic_pos + 1]  )  ;
      float x2 = (feat_ptr[basic_pos + 2] )  ;
      float y2 = (feat_ptr[basic_pos + 3] )  ;
      float x3 = (feat_ptr[basic_pos + 4] )  ;
      float y3 = (feat_ptr[basic_pos + 5])  ;
      float x4 = (feat_ptr[basic_pos + 6])  ;
      float y4 = (feat_ptr[basic_pos + 7] )  ;

    float box_objectness = feat_ptr[basic_pos + 8];
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
    {
      float box_cls_score = feat_ptr[basic_pos + 9 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold)
      {
        Object obj;
        obj.p1= cv::Point(x1,y1);
        obj.p2= cv::Point(x2,y2);
        obj.p3= cv::Point(x3,y3);
        obj.p4= cv::Point(x4,y4);
        obj.label = class_idx;
        obj.prob = box_prob;
        proposals.push_back(obj);
      }
    }
  }
//  for (auto i=proposals.begin();i!=proposals.end();i++)
//    {
//      std::cout<<i->p1<< std::endl;
//      std::cout<<i->p2<< std::endl;
//      std::cout<<i->p3<< std::endl;
//      std::cout<<i->p4<< std::endl;
//      std::cout<<i->label<< std::endl;
//      std::cout<<i->prob<< std::endl;
//      std::cout<<"*********************"<<std::endl;
//    }
}

inline float Detector::intersectionArea(const Object& a, const Object& b)
{
    cv::Rect recta;
    recta.width=abs(a.p3.x-a.p1.x);
    recta.height=abs(a.p3.y-a.p1.y);
    recta.x=(a.p3.x+a.p1.x)/2;
    recta.y=abs(a.p3.y+a.p1.y)/2;
    cv::Rect rectb;
    rectb.width=abs(b.p3.x-b.p1.x);
    rectb.height=abs(b.p3.y-b.p1.y);
    rectb.x=(b.p3.x+b.p1.x)/2;
    rectb.y=abs(b.p3.y+b.p1.y)/2;
    cv::Rect_<float> inter = recta & rectb;
    return inter.area();
}

void Detector::qsortDescentInplace(std::vector<Object>& faceobjects, int left, int right)
{
  int i = left;                                    // num on object the init one
  int j = right;                                   // num on object the last one
  float p = faceobjects[(left + right) / 2].prob;  // middle obj 's prob

  while (i <= j)
  {
    while (faceobjects[i].prob > p)
      i++;

    while (faceobjects[j].prob < p)
      j--;

    if (i <= j)
    {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }  // doing the sort work  the biggest pro obj will be seted on the init place

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j)
        qsortDescentInplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right)
        qsortDescentInplace(faceobjects, i, right);
    }
  }
}

inline void Detector::qsortDescentInplace(std::vector<Object>& proposals)
{
  if (proposals.empty())
    return;

  qsortDescentInplace(proposals, 0, proposals.size() - 1);
}

void Detector::nmsSortedBboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++)
  {
    areas[i] = abs(faceobjects[i].p3.y-faceobjects[i].p1.y)*abs(faceobjects[i].p3.x-faceobjects[i].p1.x);
  }

  for (int i = 0; i < n; i++)
  {
    const Object& a = faceobjects[i];
    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++)  //
    {
      const Object& b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area >
          nms_threshold)  // if the obj's iou is larger than nms_threshold it will be filtrated
        keep = 0;
    }

    if (keep)
      picked.push_back(i);
  }
}

void Detector::decodeOutputs(const float* prob, const int img_w, const int img_h) {
    std::vector<Object> proposals;
    generateYoloxProposals(grid_strides_, prob, bbox_conf_thresh_, proposals);  // initial filtrate

    if (proposals.empty()) {
        return;
    }
    qsortDescentInplace(proposals);
    std::vector<int> picked;
    nmsSortedBboxes(proposals, picked, nms_thresh_);
    int count = picked.size();
//    objects_.push_back(proposals[0]);
    for ( int i=0;i<count;i++)
    {
        objects_.push_back(proposals[picked[i]]);
    }
    //}
}

void Detector::drawObjects(const cv::Mat& bgr)
{
if (!objects_.empty()) {
    for (size_t i = 0; i < objects_.size(); i++) {
        cv::line(bgr, objects_[i].p1/scale_, objects_[i].p2/scale_, cv::Scalar(255, 0, 0), 2);
        cv::line(bgr, objects_[i].p2/scale_, objects_[i].p3/scale_, cv::Scalar(255, 0, 0), 2);
        cv::line(bgr, objects_[i].p3/scale_, objects_[i].p4/scale_, cv::Scalar(255, 0, 0), 2);
        cv::line(bgr, objects_[i].p1/scale_, objects_[i].p4/scale_, cv::Scalar(255, 0, 0), 2);
        cv::putText(bgr,std::to_string(objects_[i].label),objects_[i].p1/scale_,cv::FONT_HERSHEY_PLAIN,2,cv::Scalar(255,0,0));
        std::cout<<"obj1:"<<objects_[i].p1<<"obj2:"<<objects_[i].p2<<"obj3:"<<objects_[i].p3<<"obj4"<<objects_[i].p4<<std::endl;
    }
}
  camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr).toImageMsg());
}

void Detector::mainFuc(cv_bridge::CvImagePtr& image_ptr)
{
  scale_ = std::min(INPUT_W / (image_ptr->image.cols * 1.0), INPUT_H / (image_ptr->image.rows * 1.0));
  cv::Mat process_img=image_ptr->image.clone();
  staticResize(process_img);
  blobFromImage(process_img);

  infer_request_.StartAsync();
  infer_request_.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  decodeOutputs(net_pred_, process_img.cols, process_img.rows);
  if (turn_on_image_)
    drawObjects(image_ptr->image);
}

void Detector::initalizeInfer()
{
  InferenceEngine::Core ie;
  InferenceEngine::CNNNetwork network = ie.ReadNetwork(model_path_);
  std::string input_name = network.getInputsInfo().begin()->first;
  std::string output_name = network.getOutputsInfo().begin()->first;
  InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
  output_info->setPrecision(InferenceEngine::Precision::FP16);
  std::map<std::string, std::string> config = {
    { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO },
    { InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NUMA },
    { InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
      InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA },
    { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "16" }
  };
  InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU", config);
//  InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "GPU");
  InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
  infer_request_ = infer_request;
  const InferenceEngine::Blob::Ptr output_blob = infer_request_.GetBlob(output_name);
  InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
  auto moutput_holder = moutput->rmap();
  const float* net_pred =
          reinterpret_cast<const float *>(moutput_holder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type *>());
  net_pred_ = net_pred;
  InferenceEngine::Blob::Ptr img_blob = infer_request_.GetBlob(input_name);
  InferenceEngine::MemoryBlob::Ptr memory_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(img_blob);
  mblob_ = memory_blob;
}
Detector::~Detector()
{
}

}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)