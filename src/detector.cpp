
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>
#define CHECK(status)                                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    auto ret = (status);                                                                                               \
    if (ret != 0)                                                                                                      \
    {                                                                                                                  \
      std::cerr << "Cuda failure: " << ret << std::endl;                                                               \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define DEVICE 0
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int CAR_NUM_CLASSES = 1;
static const int ARMOR_NUM_CLASSES = 10;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output";
static Logger gLogger;

namespace rm_detector
{
Detector::Detector()
{
}

void Detector::onInit()
{
  nh_ = getMTPrivateNodeHandle();
  nh_.getParam("g_car_model_path", car_model_path_);
  nh_.getParam("g_armor_model_path", armor_model_path_);
  nh_.getParam("nodelet_name", nodelet_name_);
  nh_.getParam("camera_pub_name", camera_pub_name_);
  nh_.getParam("roi_data1_name", roi_data1_name_);
  nh_.getParam("roi_data2_name", roi_data2_name_);
  nh_.getParam("roi_data3_name", roi_data3_name_);
  nh_.getParam("roi_data4_name", roi_data4_name_);
  nh_.getParam("roi_data5_name", roi_data5_name_);
  nh_.getParam("left_camera", left_camera_);
  nh_.getParam("target_is_red", target_is_red_);
  nh_.getParam("target_is_blue", target_is_blue_);
  initalizeInferOfCar();
  initalizeInferOfArmor();
  ros::NodeHandle nh_reconfig(nh_, nodelet_name_ + "_reconfig");
  server_ = new dynamic_reconfigure::Server<rm_detector::dynamicConfig>(nh_reconfig);
  callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
  server_->setCallback(callback_);

  if (left_camera_)
    camera_sub_ = nh_.subscribe("/hk_camera_left/image_raw", 1, &Detector::receiveFromCam, this);
  else
    camera_sub_ = nh_.subscribe("/hk_camera_right/image_raw", 1, &Detector::receiveFromCam, this);

  camera_pub_ = nh_.advertise<sensor_msgs::Image>(nodelet_name_ + camera_pub_name_, 1);

  roi_data_pub1_ = nh_.advertise<std_msgs::Float32MultiArray>(nodelet_name_ + roi_data1_name_, 1);
  roi_data_pub2_ = nh_.advertise<std_msgs::Float32MultiArray>(nodelet_name_ + roi_data2_name_, 1);
  roi_data_pub3_ = nh_.advertise<std_msgs::Float32MultiArray>(nodelet_name_ + roi_data3_name_, 1);
  roi_data_pub4_ = nh_.advertise<std_msgs::Float32MultiArray>(nodelet_name_ + roi_data4_name_, 1);
  roi_data_pub5_ = nh_.advertise<std_msgs::Float32MultiArray>(nodelet_name_ + roi_data5_name_, 1);

  roi_data_pub_vec.push_back(roi_data_pub1_);
  roi_data_pub_vec.push_back(roi_data_pub2_);
  roi_data_pub_vec.push_back(roi_data_pub3_);
  roi_data_pub_vec.push_back(roi_data_pub4_);
  roi_data_pub_vec.push_back(roi_data_pub5_);

  red_lable_vec.push_back(0);
  red_lable_vec.push_back(1);
  red_lable_vec.push_back(2);
  red_lable_vec.push_back(3);
  red_lable_vec.push_back(4);

  blue_lable_vec.push_back(5);
  blue_lable_vec.push_back(6);
  blue_lable_vec.push_back(7);
  blue_lable_vec.push_back(8);
  blue_lable_vec.push_back(9);

  generateGridsAndStride(INPUT_W, INPUT_H);  // the wide height strides need to be changed depending on demand
}

void Detector::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
  cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
  mainFuc(cv_image_);
  car_objects_.clear();
  armor_object_vec_.clear();
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
  nms_thresh_ = config.g_nms_thresh;
  nms_thresh2_ = config.g_nms_thresh2;
  bbox_conf_thresh_ = config.g_bbox_conf_thresh;
  bbox_conf_thresh2_ = config.g_bbox_conf_thresh2;
  turn_on_image_ = config.g_turn_on_image;
  ROS_INFO("Settings have been seted");
}

void Detector::staticResize(cv::Mat& img, const float& scale)
{
  int unpad_w = scale * img.cols;
  int unpad_h = scale * img.rows;
  int resize_unpad = std::max(unpad_h, unpad_w);
  if (img.rows > img.cols)
  {
    cv::copyMakeBorder(img, img, 0, 0, abs(img.rows - img.cols) / 2, abs(img.rows - img.cols) / 2, cv::BORDER_CONSTANT,
                       cv::Scalar(122, 122, 122));
    cv::resize(img, img, cv::Size(resize_unpad, resize_unpad));
  }
  else if (img.cols > img.rows)
  {
    cv::copyMakeBorder(img, img, abs(img.rows - img.cols) / 2, abs(img.rows - img.cols) / 2, 0, 0, cv::BORDER_CONSTANT,
                       cv::Scalar(122, 122, 122));
    cv::resize(img, img, cv::Size(resize_unpad, resize_unpad));
  }
}

float* Detector::blobFromImage(cv::Mat& img)
{
  float* blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++)
  {
    for (size_t h = 0; h < img_h; h++)
    {
      for (size_t w = 0; w < img_w; w++)
      {
        blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
  return blob;
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
                                      float prob_threshold, std::vector<Object>& proposals, const int& NUM_CLASSES)
{
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
  {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

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

void Detector::publishDataForRed(const Object& object)
{
  if (object.label > 4)
    return;
  roi_data_pub_vec[object.label].publish(roi_data_);
}
void Detector::publishDataForBlue(const Object& object)
{
  if (object.label < 5)
    return;
  roi_data_pub_vec[object.label - 5].publish(roi_data_);
}

void Detector::getRoiImg(const std::vector<Object>& object, std::vector<cv::Mat>& roi_vec)
{
  for (int i = 0; i < object.size(); i++)
  {
    cv::Rect rect(object[i].rect.tl().x, object[i].rect.tl().y, object[i].rect.width, object[i].rect.height);
    if (rect.tl().x < 0)
      rect.x = 0;
    if (rect.tl().y < 0)
      rect.y = 0;
    if (rect.br().x > 640)
      rect.x = 640 - rect.width - 1;
    if (rect.br().y > 640)
      rect.y = 640 - rect.height - 1;
    cv::Mat roi = cv_image_->image(rect);
    roi_vec.push_back(roi);
  }
}

void Detector::detectArmor(std::vector<cv::Mat>& roi_vec)  // armor_point<-->roi
{
  std::vector<Object> select_car_objects;
  for (int i = 0; i < roi_vec.size(); i++)
  {
    scale2_ = std::min(INPUT_W / (roi_vec[i].cols * 1.0), INPUT_H / (roi_vec[i].rows * 1.0));
    staticResize(roi_vec[i], scale2_);
    float* blob;

    blob = blobFromImage(roi_vec[i]);

    doInference(*context2_, blob, prob2_, output_size2_, roi_vec[i].size());

    delete[] blob;

    Object armor_object;
    std::vector<Object> proposals;
    std::vector<Object> filter_objects;
    generateYoloxProposals(grid_strides_, prob2_, bbox_conf_thresh2_, proposals,
                           ARMOR_NUM_CLASSES);  // initial filtrate

    for (int j = 0; j < proposals.size(); j++)
    {
      if (target_is_red_)
      {
        auto signal = std::find(std::begin(red_lable_vec), std::end(red_lable_vec), proposals[j].label);
        if (signal != std::end(red_lable_vec))
        {
          filter_objects.push_back(proposals[j]);
        }
      }
      else if (target_is_blue_)
      {
        auto signal = std::find(std::begin(blue_lable_vec), std::end(blue_lable_vec), proposals[j].label);
        if (signal != std::end(blue_lable_vec))
        {
          filter_objects.push_back(proposals[j]);
        }
      }
    }
    proposals.assign(filter_objects.begin(), filter_objects.end());
    filter_objects.clear();

    if (proposals.empty())
    {
      cv::putText(cv_image_->image, "0", cv::Point(car_objects_[i].rect.tl().x, car_objects_[i].rect.tl().y), 1, 2,
                  cv::Scalar(255, 255, 255), 2, 2, false);
      continue;
    }
    select_car_objects.push_back(car_objects_[i]);
    qsortDescentInplace(proposals);
    std::vector<int> picked;
    nmsSortedBboxes(proposals, picked, nms_thresh2_);

    armor_object = proposals[picked[0]];
    armor_object_vec_.push_back(armor_object);
  }
  //  car_objects_ = select_car_objects;
  car_objects_.assign(select_car_objects.begin(), select_car_objects.end());
  select_car_objects.clear();
}

void Detector::drawObjects(const cv::Mat& bgr)
{
  if (target_is_red_)
    for (size_t i = 0; i < car_objects_.size(); i++)
    {
      cv::putText(cv_image_->image, std::to_string(armor_object_vec_[i].label + 1),
                  cv::Point(car_objects_[i].rect.x, car_objects_[i].rect.y), 1, 2, cv::Scalar(0, 0, 255), 2, 2, false);
      cv::rectangle(bgr, car_objects_[i].rect, cv::Scalar(0, 255, 0), 2);
    }
  if (target_is_blue_)
    for (size_t i = 0; i < car_objects_.size(); i++)
    {
      cv::putText(cv_image_->image, std::to_string(armor_object_vec_[i].label - 4),
                  cv::Point(car_objects_[i].rect.x, car_objects_[i].rect.y), 1, 2, cv::Scalar(255, 0, 0), 2, 2, false);
      cv::rectangle(bgr, car_objects_[i].rect, cv::Scalar(0, 255, 0), 2);
    }

  camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr).toImageMsg());
}

void Detector::doInference(nvinfer1::IExecutionContext& context, float* input, float* output, const int output_size,
                           const cv::Size& input_shape)
{
  const nvinfer1::ICudaEngine& engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void* buffers[2];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int input_index = engine.getBindingIndex(INPUT_BLOB_NAME);

  assert(engine.getBindingDataType(input_index) == nvinfer1::DataType::kFLOAT);
  const int output_index = engine.getBindingIndex(OUTPUT_BLOB_NAME);
  assert(engine.getBindingDataType(output_index) == nvinfer1::DataType::kFLOAT);
  //  int mBatchSize = engine.getMaxBatchSize();

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[input_index], 3 * input_shape.height * input_shape.width * sizeof(float)));
  CHECK(cudaMalloc(&buffers[output_index], output_size * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  CHECK(cudaMemcpyAsync(buffers[input_index], input, 3 * input_shape.height * input_shape.width * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(1, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[output_index], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[input_index]));
  CHECK(cudaFree(buffers[output_index]));
}

void Detector::initalizeInferOfCar()
{
  cudaSetDevice(DEVICE);
  // create a model using the API directly and serialize it to a stream
  char* trt_model_stream{ nullptr };
  size_t size{ 0 };

  //  if (argc == 4 && std::string(argv[2]) == "-i") {
  const std::string engine_file_path = car_model_path_;
  std::ifstream file(engine_file_path, std::ios::binary);
  if (file.good())
  {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    assert(trt_model_stream);
    file.read(trt_model_stream, size);
    file.close();
  }

  trt_model_stream_ = trt_model_stream;

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  runtime_ = runtime;
  assert(runtime_ != nullptr);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream_, size);
  engine_ = engine;
  assert(engine_ != nullptr);
  nvinfer1::IExecutionContext* context = engine_->createExecutionContext();
  context_ = context;
  assert(context_ != nullptr);
  delete[] trt_model_stream;
  auto out_dims = engine_->getBindingDimensions(1);
  auto output_size = 1;
  for (int j = 0; j < out_dims.nbDims; j++)
  {
    output_size *= out_dims.d[j];
  }
  output_size_ = output_size;

  static float* prob = new float[output_size_];
  prob_ = prob;
}

void Detector::initalizeInferOfArmor()
{
  cudaSetDevice(DEVICE);
  // create a model using the API directly and serialize it to a stream
  char* trt_model_stream{ nullptr };
  size_t size{ 0 };

  //  if (argc == 4 && std::string(argv[2]) == "-i") {
  const std::string engine_file_path = armor_model_path_;
  std::ifstream file(engine_file_path, std::ios::binary);
  if (file.good())
  {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    assert(trt_model_stream);
    file.read(trt_model_stream, size);
    file.close();
  }

  trt_model_stream2_ = trt_model_stream;

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  runtime2_ = runtime;
  assert(runtime2_ != nullptr);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream2_, size);
  engine2_ = engine;
  assert(engine2_ != nullptr);
  nvinfer1::IExecutionContext* context = engine2_->createExecutionContext();
  context2_ = context;
  assert(context2_ != nullptr);
  delete[] trt_model_stream;
  auto out_dims = engine2_->getBindingDimensions(1);
  auto output_size = 1;
  for (int j = 0; j < out_dims.nbDims; j++)
  {
    output_size *= out_dims.d[j];
  }
  output_size2_ = output_size;

  static float* prob = new float[output_size2_];
  prob2_ = prob;
}

void Detector::mainFuc(cv_bridge::CvImagePtr& image_ptr)
{
  scale_ = std::min(INPUT_W / (cv_image_->image.cols * 1.0), INPUT_H / (cv_image_->image.rows * 1.0));
  int img_w = cv_image_->image.cols;
  int img_h = cv_image_->image.rows;
  staticResize(cv_image_->image, scale_);

  float* blob;
  blob = blobFromImage(cv_image_->image);

  doInference(*context_, blob, prob_, output_size_, cv_image_->image.size());

  delete[] blob;

  std::vector<Object> proposals;
  generateYoloxProposals(grid_strides_, prob_, bbox_conf_thresh_, proposals, CAR_NUM_CLASSES);  // initial filtrate
                                                                                                //  ROS_INFO("finding");
  //  if (proposals.empty())
  //  {
  //    return;
  //  }
  //  ROS_INFO("find car!");

  camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
  qsortDescentInplace(proposals);
  std::vector<int> picked;
  nmsSortedBboxes(proposals, picked, nms_thresh_);

  int count = picked.size();
  car_objects_.resize(count);

  for (int i = 0; i < count; i++)
  {
    car_objects_[i] = proposals[picked[i]];

    float x0 = (car_objects_[i].rect.x);
    float y0 = (car_objects_[i].rect.y);
    float x1 = (car_objects_[i].rect.x + car_objects_[i].rect.width);
    float y1 = (car_objects_[i].rect.y + car_objects_[i].rect.height);

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    car_objects_[i].rect.x = x0;
    car_objects_[i].rect.y = y0;
    car_objects_[i].rect.width = x1 - x0;
    car_objects_[i].rect.height = y1 - y0;
  }  // make the real object

  std::vector<cv::Mat> roi_vec;
  getRoiImg(car_objects_, roi_vec);  // obj->roi->armor
  detectArmor(roi_vec);

  if (armor_object_vec_.empty())
    return;

  ROS_INFO("find armor!");
  if (car_objects_.size() > 5)
    car_objects_.resize(5);

  //  std::vector<int> detectable_num_vec;
  for (size_t i = 0; i < car_objects_.size(); i++)
  {
    roi_point_vec_.clear();
    roi_data_.data.clear();

    roi_data_point_l_.x = (car_objects_[i].rect.tl().x) / scale_;
    roi_data_point_l_.y = ((car_objects_[i].rect.tl().y) / scale_) - (abs(img_w - img_h) / 2);
    roi_data_point_r_.x = (car_objects_[i].rect.br().x) / scale_;
    roi_data_point_r_.y = ((car_objects_[i].rect.br().y) / scale_) - (abs(img_w - img_h) / 2);

    roi_point_vec_.push_back(roi_data_point_l_);
    roi_point_vec_.push_back(roi_data_point_r_);

    roi_data_.data.push_back(roi_point_vec_[0].x);  // output the point in origin img
    roi_data_.data.push_back(roi_point_vec_[0].y);
    roi_data_.data.push_back(roi_point_vec_[1].x);
    roi_data_.data.push_back(roi_point_vec_[1].y);

    // add camera message
    if (left_camera_)
      roi_data_.data.push_back(0);
    else
      roi_data_.data.push_back(1);

    if (target_is_red_)
    {
      publishDataForRed(armor_object_vec_[i]);
    }
    else if (target_is_blue_)
    {
      publishDataForBlue(armor_object_vec_[i]);
    }
    //    detectable_num_vec.push_back(armor_object_vec_[i].label);
  }
  //  if (!prev_objects_.empty())
  //  {
  //    if (target_is_red_)
  //      publishUndetectableNum(detectable_num_vec, red_lable_vec, prev_objects_, img_w, img_h);
  //    else if (target_is_blue_)
  //      publishUndetectableNum(detectable_num_vec, blue_lable_vec, prev_objects_, img_w, img_h);
  //  }
  if (turn_on_image_)
    drawObjects(cv_image_->image);
}

Detector::~Detector()
{
}
}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)