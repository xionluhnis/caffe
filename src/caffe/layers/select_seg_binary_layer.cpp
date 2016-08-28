#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/select_seg_binary_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype, int NumImages>
SelectSegBinaryLayer<Dtype, NumImages>::~SelectSegBinaryLayer<Dtype, NumImages>() {
  this->StopInternalThread();
}

template <typename Dtype, int NumImages>
void SelectSegBinaryLayer<Dtype, NumImages>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  // const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "SelectSegBinaryLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  label_dim_ = this->layer_param_.window_cls_data_param().label_dim();

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    SegItems item;
    for(unsigned int i = 0; i < NumImages; ++i) {
      string imgfn;
      iss >> imgfn;
      item.imgfn[i] = imgfn;
    }

    int x1, y1, x2, y2;
    iss >> x1 >> y1 >> x2 >> y2;
    item.x1 = x1;
    item.y1 = y1;
    item.x2 = x2;
    item.y2 = y2;

    for (int i = 0; i < label_dim_; i++) {
      int l;
      iss >> l;
      item.cls_label.push_back(l);
    }

    lines_.push_back(item);
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  for(unsigned int i = 0; i < NumImages; ++i){
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].imgfn[i],
                                      new_height, new_width, is_color && i != NumImages - 1);
    // image
    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    xformed_data_[i].Reshape(top_shape);
    // Reshape prefetch_data and top[0] according to the batch_size.
    top_shape[0] = batch_size;
    for (int j = 0; j < this->PREFETCH_COUNT; ++j) {
      this->prefetch_[j].data_[i].Reshape(top_shape);
    }
    top[i]->Reshape(top_shape);

    LOG(INFO) << "output data (" << i << ") size: " << top[i]->num() << ","
        << top[i]->channels() << "," << top[i]->height() << ","
        << top[i]->width();
  
  }

  // class label
  top[NumImages]->Reshape(batch_size, label_dim_, 1, 1);
  this->prefetch_data_dim_.Reshape(batch_size, label_dim_, 1, 1);
  this->class_label_.Reshape(1, label_dim_, 1, 1);
  
  LOG(INFO) << "output class label size: " << top[NumImages]->num() << ","
	    << top[NumImages]->channels() << "," << top[NumImages]->height() << ","
	    << top[NumImages]->width();
}

template <typename Dtype, int NumImages>
void SelectSegBinaryLayer<Dtype, NumImages>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype, int NumImages>
void SelectSegBinaryLayer<Dtype, NumImages>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  for(unsigned int i = 0; i < NumImages; ++i){
    CHECK(batch->data_[i].count());
    CHECK(xformed_data_[i].count());
  }
  
  Dtype* top_cls_label = this->prefetch_data_dim_.mutable_cpu_data();

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  
  // const int label_type = this->layer_param_.image_data_param().label_type();
  const int ignore_label = image_data_param.ignore_label();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // retrieve the blobs
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    
    std::vector<cv::Mat> cv_img_list;
    cv_img_list.reserve(NumImages);
    for (int i = 0; i < NumImages; ++i) {
      // load image
      cv::Mat cv_img;
      if(i < NumImages - 1) {
        cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].imgfn[i],
              0, 0, is_color);
      } else {
        cv_img = ReadImageToCVMatNearest(root_folder + lines_[lines_id_].imgfn[i],
					    0, 0, false);
      }

      CHECK(cv_img.data) << "Could not load img_" << i << ": " << lines_[lines_id_].imgfn[i];
      read_time += timer.MicroSeconds();
      timer.Start();
      
      // crop window out of image and warp it
      int x1 = lines_[lines_id_].x1;
      int y1 = lines_[lines_id_].y1;
      int x2 = lines_[lines_id_].x2;
      int y2 = lines_[lines_id_].y2;
      // compute padding 
      int pad_x1 = std::max(0, -x1);
      int pad_y1 = std::max(0, -y1);
      int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
      int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
      if (pad_x1 > 0 || pad_x2 > 0 || pad_y1 > 0 || pad_y2 > 0) {
        cv::Scalar color(0,0,0);
        if(i + 1 == NumImages) // special case for label
          color = cv::Scalar(ignore_label);
        cv::copyMakeBorder(cv_img, cv_img, pad_y1, pad_y2,
            pad_x1, pad_x2, cv::BORDER_CONSTANT, color); 
      }
      // clip bounds
      x1 = x1 + pad_x1;
      x2 = x2 + pad_x1;
      y1 = y1 + pad_y1;
      y2 = y2 + pad_y1;
      CHECK_GT(x1, -1);
      CHECK_GT(y1, -1);
      CHECK_LT(x2, cv_img.cols);
      CHECK_LT(y2, cv_img.rows);
   
      // cropping
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      if (new_width > 0 && new_height > 0) {
        cv::resize(cv_cropped_img, cv_cropped_img, 
               cv::Size(new_width, new_height), 0, 0, i == NumImages - 1 ? cv::INTER_LINEAR : cv::INTER_NEAREST);
      }
      
      cv_img_list.push_back(cv_cropped_img);
      
      // Note: already done in DataLayerSetUp
      // Use data_transformer to infer the expected blob shape from a cv_image.
      // vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
      // xformed_data_[i].Reshape(top_shape);
    
      // target after transformation
      Dtype* prefetch_data = batch->data_[i].mutable_cpu_data();
      int offset = batch->data_[i].offset(item_id);
      // Reshape transformed data
      xformed_data_[i].set_cpu_data(prefetch_data + offset);
    }

    timer.Start();
    // Apply transformations (mirror, crop...) to the images
    this->data_transformer_->TransformImgs(cv_img_list, &xformed_data_[0], ignore_label);
    trans_time += timer.MicroSeconds();

    // class label
    int offset = this->prefetch_data_dim_.offset(item_id);
    this->class_label_.set_cpu_data(top_cls_label + offset);
    Dtype * cls_label_data = this->class_label_.mutable_cpu_data();
    for (int i = 0; i < label_dim_; i++) {
      cls_label_data[i] = lines_[lines_id_].cls_label[i];
    }

    // modify seg label
    Dtype * seg_label_data = xformed_data_[NumImages - 1].mutable_cpu_data();
    int pixel_count = xformed_data_[NumImages - 1].count();
    const int cls_label_base = this->layer_param_.select_seg_binary_param().cls_label_base();
    for (int i = 0; i < pixel_count; i++) {
      int seg_label = seg_label_data[i];
      if (seg_label != 0 && seg_label != ignore_label) {
        if (cls_label_base < seg_label && seg_label-1 < (label_dim_+cls_label_base)) {
          seg_label_data[i] = cls_label_data[seg_label-cls_label_base-1];
        }
        else {
          seg_label_data[i] = 0;
        }
      }
    }

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	      ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#define GEN_LAYER_IMPL(n, suffix, className, instName) \
  template< typename Dtype> \
  class className : public SelectSegBinaryLayer<Dtype, n> { \
  public: \
    explicit className(const LayerParameter& param) : SelectSegBinaryLayer<Dtype, n>(param) {} \
    ~className() { this->StopInternalThread(); } \
    virtual inline const char* type() const { return "SelectSegBinary" #n; } \
  }; \
  INSTANTIATE_CLASS(className); \
  REGISTER_LAYER_CLASS(instName);
    
#define GEN_LAYER_IMPL_X(n, suffix, className, instName) GEN_LAYER_IMPL(n, suffix, className, instName)
#define GEN_LAYER(n, suffix) GEN_LAYER_IMPL_X(n, suffix, SelectSegBinary ## suffix  ## Layer, SelectSegBinary ## suffix)

// generate implementations for all interesting numbers of frames
GEN_LAYER(1, OneFrame);
// GEN_LAYER(2, TwoFrames);
// GEN_LAYER(3, ThreeFrames);
// GEN_LAYER(4, FourFrames);
// GEN_LAYER(5, FiveFrames);
// GEN_LAYER(6, SixFrames);
// GEN_LAYER(7, SevenFrames);
// GEN_LAYER(8, EightFrames);

}  // namespace caffe
