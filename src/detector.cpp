
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>
#define INPUT_W 640
#define INPUT_H 640
#define NUM_CLASSES 1
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
  nh.getParam("distortion_coefficients/data", discoeffs_vec_);
  nh.getParam("camera_matrix/data", camera_matrix_vec_);
  nh.getParam("nodelet_name", nodelet_name_);
  nh.getParam("camera_pub_name", camera_pub_name_);
  nh.getParam("roi_data1_name", roi_data1_name_);
  nh.getParam("roi_data2_name", roi_data2_name_);
  nh.getParam("roi_data3_name", roi_data3_name_);
  nh.getParam("roi_data4_name", roi_data4_name_);
  nh.getParam("roi_data5_name", roi_data5_name_);
//  setDataToMatrix(discoeffs_vec_, camera_matrix_vec_);
    initalizeInfer();
    callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
    server_.setCallback(callback_);

    nh_ = ros::NodeHandle(nh, nodelet_name_);
  camera_sub_ = nh_.subscribe("/galaxy_camera/image_raw", 1, &Detector::receiveFromCam, this);
  camera_pub_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_, 1);
  camera_pub2_ = nh_.advertise<sensor_msgs::Image>("sub_publisher", 1);

  roi_data_pub1_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data1_name_, 1);
  roi_data_pub2_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data2_name_, 1);
  roi_data_pub3_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data3_name_, 1);
  roi_data_pub4_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data4_name_, 1);
  roi_data_pub5_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data5_name_, 1);

  roi_data_pub_vec.push_back(roi_data_pub1_);
  roi_data_pub_vec.push_back(roi_data_pub2_);
  roi_data_pub_vec.push_back(roi_data_pub3_);
  roi_data_pub_vec.push_back(roi_data_pub4_);
  roi_data_pub_vec.push_back(roi_data_pub5_);

  generateGridsAndStride(INPUT_W, INPUT_H);  // the wide height strides need to be changed depending on demand
}

void Detector::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
  cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
  mainFuc(cv_image_);
  objects_.clear();
  roi_picture_vec_.clear();
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
  nms_thresh_ = config.g_nms_thresh;
  bbox_conf_thresh_ = config.g_bbox_conf_thresh;
  turn_on_image_ = config.g_turn_on_image;
  undistort_bool_ = config.g_undistort_bool;
  target_is_red_ = config.target_is_red;
  target_is_blue_ = config.target_is_blue;
  ratio_of_pixels_ = config.ratio_of_pixels;
  pixels_thresh_ = config.pixels_thresh;
  ROS_INFO("Settings have been seted");
}

void Detector::setDataToMatrix(std::vector<float> disc_vec, std::vector<float> cam_vec)
{
  discoeffs_ = (cv::Mat_<float>(1, 5) << disc_vec[0], disc_vec[1], disc_vec[2], disc_vec[3], disc_vec[4]);
  camera_matrix_ = (cv::Mat_<float>(3, 3) << cam_vec[0], cam_vec[1], cam_vec[2], cam_vec[3], cam_vec[4], cam_vec[5],
                    cam_vec[6], cam_vec[7], cam_vec[8]);
}

void Detector::staticResize(cv::Mat& img)
{
  // r = std::min(r, 1.0f);
  int unpad_w = scale_ * img.cols;
  int unpad_h = scale_ * img.rows;
  int resize_unpad = std::max(unpad_h, unpad_w);
  //  auto origin_img_ptr = std::make_shared<cv::Mat>(img);
  origin_img_w_ = img.cols;
  origin_img_h_ = img.rows;
  cv::copyMakeBorder(img, img, abs(img.rows - img.cols) / 2, abs(img.rows - img.cols) / 2, 0, 0, cv::BORDER_CONSTANT,
                     cv::Scalar(122, 122, 122));
  cv::resize(img, img, cv::Size(resize_unpad, resize_unpad));
  //  cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
  //  img.copyTo(out(cv::Rect(0, 0, img.cols, img.rows)));
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
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

    // yolox/models/yolo_head.py decode logic
    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
    float w = exp(feat_ptr[basic_pos + 2]) * stride;
    float h = exp(feat_ptr[basic_pos + 3]) * stride;
    // above all get from model
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_ptr[basic_pos + 4];
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
    {
      float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold)
      {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        //                    obj.lu=cv::Point2f (x0-(w/2.0),y0+(h/2.0));
        //                    obj.ld=cv::Point2f (x0-(w/2.0),y0-(h/2.0));
        //                    obj.ru=cv::Point2f (x0+(w/2.0),y0+(h/2.0));
        //                    obj.rd=cv::Point2f (x0+(w/2.0),y0-(h/2.0));
        obj.label = class_idx;
        obj.prob = box_prob;

        proposals.push_back(obj);
      }
    }
  }
}

inline float Detector::intersectionArea(const Object& a, const Object& b)
{
  cv::Rect_<float> inter = a.rect & b.rect;
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
    areas[i] = faceobjects[i].rect.area();
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

void Detector::selectTargetColor(std::vector<Object>& proposals)
{

    for (int i = 0; i < proposals.size(); i++)
  {
    cv::Rect rect(proposals[i].rect.tl().x, proposals[i].rect.tl().y, proposals[i].rect.width, proposals[i].rect.height);
    if (rect.tl().x < 0)
      rect.x = 0;
    if (rect.br().x > 640)
      rect.x = cv_image_->image.cols - rect.width;
    if (rect.tl().y < 0)
      rect.y = 0;
    if (rect.br().y > 640)
      rect.y = cv_image_->image.rows - rect.height;
    roi_picture_ = cv_image_->image(rect);
    roi_picture_vec_.push_back(roi_picture_);
  }
  std::vector<Object> filter_objects;

  // select the target's color
  if (target_is_red_)
  {
    for (int i = 0; i < roi_picture_vec_.size(); i++)
    {
        cv::split(roi_picture_vec_[i],roi_picture_split_vec_);
        cv::Mat r_decrease_b; //r-b
        cv::subtract(roi_picture_split_vec_[2],roi_picture_split_vec_[0],r_decrease_b);

        counter_of_pixel_=0;

      for (int j = 0; j < r_decrease_b.cols; j++)
      {
        for (int k = 0; k < r_decrease_b.rows; k++)
        {
          if (r_decrease_b.at<uchar>(j, k) >pixels_thresh_)
            counter_of_pixel_++;
        }
      }

      camera_pub2_.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", r_decrease_b).toImageMsg());
      if (counter_of_pixel_ / (r_decrease_b.rows * r_decrease_b.cols) > ratio_of_pixels_)
        filter_objects.push_back(proposals[i]);
      roi_picture_split_vec_.clear();

    }
    proposals.assign(filter_objects.begin(), filter_objects.end());
  }

  else if (target_is_blue_)
  {
    for (int i = 0; i < roi_picture_vec_.size(); i++)
    {
        cv::split(roi_picture_vec_[i],roi_picture_split_vec_);
        cv::Mat b_decrease_r;
        cv::subtract(roi_picture_split_vec_[0],roi_picture_split_vec_[2],b_decrease_r); //b-r

        counter_of_pixel_=0;
        for (int j = 0; j < b_decrease_r.cols; j++)
      {
        for (int k = 0; k < b_decrease_r.rows; k++)
        {
          if (b_decrease_r.at<uchar>(j, k) >pixels_thresh_)
            counter_of_pixel_++;
        }
      }
      camera_pub2_.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", b_decrease_r).toImageMsg());
      if (counter_of_pixel_ / (b_decrease_r.rows * b_decrease_r.cols) > ratio_of_pixels_)
        filter_objects.push_back(proposals[i]);
      roi_picture_split_vec_.clear();
    }
    proposals.assign(filter_objects.begin(), filter_objects.end());
  }
}

void Detector::decodeOutputs(const float* prob, const int img_w, const int img_h)
{
  std::vector<Object> proposals;
  generateYoloxProposals(grid_strides_, prob, bbox_conf_thresh_, proposals);  // initial filtrate

  selectTargetColor(proposals);

  if (proposals.empty()) {
      return;
  }

  qsortDescentInplace(proposals);
  std::vector<int> picked;
  nmsSortedBboxes(proposals, picked, nms_thresh_);


  int count = picked.size();
  if (count > 5)
    count = 5;
  objects_.resize(count);

  for (int i = 0; i < count; i++)
  {
    objects_[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    //    float x0 = (objects_[i].rect.x) / scale;
    //    float y0 = (objects_[i].rect.y) / scale;
    //    float x1 = (objects_[i].rect.x + objects_[i].rect.width) / scale;
    //    float y1 = (objects_[i].rect.y + objects_[i].rect.height) / scale;

    float x0 = (objects_[i].rect.x);
    float y0 = (objects_[i].rect.y);
    float x1 = (objects_[i].rect.x + objects_[i].rect.width);
    float y1 = (objects_[i].rect.y + objects_[i].rect.height);

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    objects_[i].rect.x = x0;
    objects_[i].rect.y = y0;
    objects_[i].rect.width = x1 - x0;
    objects_[i].rect.height = y1 - y0;
  }  // make the real object
     //  for (size_t i = 0; i < objects_.size(); i++)


  for (size_t i = 0; i < 5; i++)
  {
    roi_point_vec_.clear();
    roi_data_.data.clear();

    roi_data_point_l_.x = (objects_[i].rect.tl().x) / scale_;
    roi_data_point_l_.y = ((objects_[i].rect.tl().y) / scale_) - (abs(origin_img_w_ - origin_img_h_) / 2);
    roi_data_point_r_.x = (objects_[i].rect.br().x) / scale_;
    roi_data_point_r_.y = ((objects_[i].rect.br().y) / scale_) - (abs(origin_img_w_ - origin_img_h_) / 2);

    roi_point_vec_.push_back(roi_data_point_l_);
    roi_point_vec_.push_back(roi_data_point_r_);

    if (undistort_bool_)
    {
      cv::undistortPoints(roi_point_vec_, roi_point_vec_, camera_matrix_, discoeffs_, cv::noArray());
    }

    roi_data_.data.push_back(roi_point_vec_[0].x);
    roi_data_.data.push_back(roi_point_vec_[0].y);
    roi_data_.data.push_back(roi_point_vec_[1].x);
    roi_data_.data.push_back(roi_point_vec_[1].y);
    roi_data_pub_vec[i].publish(roi_data_);
  }
}

void Detector::drawObjects(const cv::Mat& bgr)
{
  //        static const char* class_names[] = {
  //                "armor"
  //        };

  for (size_t i = 0; i < objects_.size(); i++)
  {
    //        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
    //                obj.rect.tl().x, obj.rect.tl().y, obj.rect.width, obj.rect.height);

    //        float c_mean = cv::mean(color)[0];
    //        cv::Scalar txt_color;
    //        if (c_mean > 0.5){
    //            txt_color = cv::Scalar(0, 0, 0);
    //        }else{
    //            txt_color = cv::Scalar(255, 255, 255);
    //        }

    cv::rectangle(bgr, objects_[i].rect, cv::Scalar(255, 0, 0), 2);

    //        char text[256];
    //        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
    //
    //        int baseLine = 0;
    //        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
    //
    //        cv::Scalar txt_bk_color = color * 0.7 * 255;
    //
    //        int x = obj.rect.x;
    //        int y = obj.rect.y + 1;
    //        //int y = obj.rect.y - label_size.height - baseLine;
    //        if (y > image.rows)
    //            y = image.rows;
    //        //if (x + label_size.width > image.cols)
    //        //x = image.cols - label_size.width;
    //
    //        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
    //                      txt_bk_color, -1);
    //
    //        cv::putText(image, text, cv::Point(x, y + label_size.height),
    //                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }
  camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr).toImageMsg());
}

void Detector::mainFuc(cv_bridge::CvImagePtr& image_ptr)
{
  scale_ = std::min(INPUT_W / (image_ptr->image.cols * 1.0), INPUT_H / (image_ptr->image.rows * 1.0));
  staticResize(image_ptr->image);
  blobFromImage(image_ptr->image);

  infer_request_.StartAsync();
  infer_request_.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  decodeOutputs(net_pred_, image_ptr->image.cols, image_ptr->image.rows);
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
  output_info->setPrecision(InferenceEngine::Precision::FP32);
  std::map<std::string, std::string> config = {
    { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO },
    { InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NUMA },
    { InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
      InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA },
    { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "16" }
  };
  InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU", config);
  InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
  infer_request_ = infer_request;
  const InferenceEngine::Blob::Ptr output_blob = infer_request_.GetBlob(output_name);
  InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
  auto moutput_holder = moutput->rmap();
  const float* net_pred =
      moutput_holder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
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