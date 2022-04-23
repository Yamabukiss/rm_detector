
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>
#include "fstream"
#define INPUT_W  640
#define INPUT_H  640
#define NUM_CLASSES  1
float g_nms_thresh=0.1;
float g_bbox_conf_thresh=0.1;
std_msgs::Int16MultiArray g_roi_data;

namespace rm_detector
{
static bool REPEAT{};
Detector::Detector()
{

}

void Detector::initialize( const ros::NodeHandle& nh)
{
  Detector::nh_ = ros::NodeHandle(nh, "detector_node");
  camera_sub_=nh_.subscribe("/galaxy_camera/image_raw",1,&Detector::receiveFromCam,this);
  camera_pub_=nh_.advertise<sensor_msgs::Image>("armor_detect",1);
//  Detector::it_ = std::make_shared<image_transport::ImageTransport>(nh_);

//  camera_sub_ = it_->subscribeCamera(nh_.param("camera_name", std::string("/galaxy_camera")) + "/image_raw", 20,
//                                     &Detector::receiveFromCam,this);
//  camera_pub_=it_->advertise("armor_detect",1);

  roi_data_pub1_=nh_.advertise<std_msgs::Int16MultiArray>("roi_data1",1);
  roi_data_pub2_=nh_.advertise<std_msgs::Int16MultiArray>("roi_data2",1);
  roi_data_pub3_=nh_.advertise<std_msgs::Int16MultiArray>("roi_data3",1);

  roi_data_pub_vec.push_back(roi_data_pub1_);
  roi_data_pub_vec.push_back(roi_data_pub1_);
  roi_data_pub_vec.push_back(roi_data_pub1_);

  generateGridsAndStride(INPUT_W, INPUT_H); // the wide height strides need to be changed depending on demand


}

 void Detector::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{

  try
  {
    Detector::cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
    REPEAT = true;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }


}


 bool Detector::getCam(cv_bridge::CvImagePtr &cv_image, sensor_msgs::CameraInfoConstPtr &cam_info) {
  if (cv_image_ == nullptr)
    return false;
  if (!Detector::cv_image_->image.empty() && REPEAT) {
    cv_image = Detector::cv_image_;
    REPEAT = false;
    return true;
  } else
    return false;
}


     void Detector::sendMsg(boost::shared_ptr<sensor_msgs::Image> msg)
    {
        camera_pub_.publish(msg);
    }

//    inline  void Detector::sendROI(std_msgs::Int16MultiArray roi_data)
//    {
//        roi_data_pub_.publish(roi_data);
//    }

     cv::Mat Detector::staticResize(cv::Mat& img) {
        float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        // r = std::min(r, 1.0f);
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows;
        cv::resize(img, img, cv::Size(unpad_w,unpad_h));
        cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
        img.copyTo(out(cv::Rect(0, 0, img.cols, img.rows)));
        return out;
    }

    void Detector::blobFromImage(cv::Mat& img, InferenceEngine::MemoryBlob::Ptr &mblob){
        int channels = 3;
        int img_h = img.rows;
        int img_w = img.cols;
        if (!mblob)
        {
            THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                               << "but by fact we were not able to cast inputBlob to MemoryBlob";
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto mblobHolder = mblob->wmap();

        float *blob_data = mblobHolder.as<float *>();

        for (size_t c = 0; c < channels; c++)
        {
            for (size_t  h = 0; h < img_h; h++)
            {
                for (size_t w = 0; w < img_w; w++)
                {
                    blob_data[c * img_w * img_h + h * img_w + w] =
                            (float)img.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }

// if the 3 parameters is fixed the strides can be fixed too
     void Detector::generateGridsAndStride(const int target_w, const int target_h)
    {
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        for (auto stride : strides)
        {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;
            for (int g1 = 0; g1 < num_grid_h; g1++)
            {
                for (int g0 = 0; g0 < num_grid_w; g0++)
                {
                    grid_strides.push_back((GridAndStride){g0, g1, stride});
                }
            }
        }

        Detector::grid_strides_=grid_strides;
    }

     void Detector::generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects)
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

                    objects.push_back(obj);
                }

            } // class loop

        } // point anchor loop
    }

     inline float Detector::intersectionArea(const Object& a, const Object& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

     void Detector::qsortDescentInplace(std::vector<Object>& faceobjects, int left, int right)
    {
        int i = left; // num on object the init one
        int j = right; // num on object the last one
        float p = faceobjects[(left + right) / 2].prob; // middle obj 's prob

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
        }  //doing the sort work  the biggest pro obj will be seted on the init place

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j) qsortDescentInplace(faceobjects, left, j);
            }
#pragma omp section
            {
                if (i < right) qsortDescentInplace(faceobjects, i, right);
            }
        }
    }

     void Detector::qsortDescentInplace(std::vector<Object>& objects)
    {
        if (objects.empty())
            return;

        qsortDescentInplace(objects, 0, objects.size() - 1);
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
            for (int j = 0; j < (int)picked.size(); j++) //
            {
                const Object& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersectionArea(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold) //if the obj's iou is larger than nms_threshold it will be filtrated
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

     void Detector::decodeOutputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        generateYoloxProposals(Detector::grid_strides_, prob,  g_bbox_conf_thresh, proposals); // initial filtrate
         if (proposals.empty()) return;
        qsortDescentInplace(proposals);

        std::vector<int> picked;
        nmsSortedBboxes(proposals, picked, g_nms_thresh);
        int count = picked.size();
        objects.resize(count);

        for (int i = 0; i < count; i++)
        {
//        objects[i] = proposals[picked[i]];
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        } // make the real object
    }

    sensor_msgs::ImagePtr Detector::drawObjects(const cv::Mat& bgr, const std::vector<Object>& objects)
    {
//        static const char* class_names[] = {
//                "armor"
//        };


        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            g_roi_data.data.clear();
            g_roi_data.data.push_back(obj.rect.tl().x);
            g_roi_data.data.push_back(obj.rect.tl().y);
            g_roi_data.data.push_back(obj.rect.width);
            g_roi_data.data.push_back(obj.rect.height);
            roi_data_pub_vec[i].publish(g_roi_data);
//            sendROI(g_roi_data);
//        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                obj.rect.tl().x, obj.rect.tl().y, obj.rect.width, obj.rect.height);

//        float c_mean = cv::mean(color)[0];
//        cv::Scalar txt_color;
//        if (c_mean > 0.5){
//            txt_color = cv::Scalar(0, 0, 0);
//        }else{
//            txt_color = cv::Scalar(255, 255, 255);
//        }

            cv::rectangle(bgr, obj.rect, cv::Scalar (255,0,0), 3);

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
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr).toImageMsg();
        return msg;
    }

    sensor_msgs::ImagePtr Detector::mainFuc(cv_bridge::CvImagePtr& image_ptr,std::vector<Object> objects) {

        cv::Mat pr_img = staticResize(image_ptr->image);
        blobFromImage(pr_img, Detector::mblob_);

        Detector::infer_request_.StartAsync();
        Detector::infer_request_.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

        int img_w = image_ptr->image.cols;
        int img_h = image_ptr->image.rows;
        float scale = std::min(INPUT_W / (image_ptr->image.cols*1.0), INPUT_H / (image_ptr->image.rows*1.0));

        decodeOutputs(Detector::net_pred_, objects, scale, img_w, img_h);
        return drawObjects(image_ptr->image, objects);
    }

    void  Detector::initalize_infer()
    {
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork network = ie.ReadNetwork("../armor3.onnx");
        std::string input_name = network.getInputsInfo().begin()->first;
        std::string output_name = network.getOutputsInfo().begin()->first;
        InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
        output_info->setPrecision(InferenceEngine::Precision::FP32);
        std::map<std::string, std::string> config = {{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,InferenceEngine::PluginConfigParams::NO},
                                                     {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD,InferenceEngine::PluginConfigParams::NUMA},
                                                     {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA},
                                                     {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM,"16"}};
        InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU",config);
        InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
        Detector::infer_request_=infer_request;

        const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(output_name);
        InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
        auto moutputHolder = moutput->rmap();
        const float* net_pred = moutputHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        Detector::net_pred_=net_pred;
        InferenceEngine::Blob::Ptr imgBlob = infer_request.GetBlob(input_name);
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(imgBlob);
        Detector::mblob_=mblob;
    }
    Detector::~Detector()
{
}
InferenceEngine::InferRequest Detector::infer_request_;
InferenceEngine::MemoryBlob::Ptr Detector::mblob_;
const float* Detector::net_pred_;
cv_bridge::CvImagePtr Detector::cv_image_;
std::vector<GridAndStride> Detector::grid_strides_;

}  // namespace rm_detector