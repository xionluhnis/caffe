#ifndef SELECT_SEG_BINARY_TWO_FRAMES_HPP
#define SELECT_SEG_BINARY_TWO_FRAMES_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_dim_prefetching_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SelectSegBinaryTwoFramesLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit SelectSegBinaryTwoFramesLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~SelectSegBinaryTwoFramesLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SelectSegBinaryTwoFrames"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 4; }
  virtual inline bool AutoTopBlobs() const { return true; }
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

 protected:
  Blob<Dtype> transformed_label_;
  Blob<Dtype> class_label_;
  
  Blob<Dtype> prefetch_data2_;
  Blob<Dtype> transformed_data2_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  typedef struct SegItems {
    std::string img1fn;
    std::string img2fn;
    std::string segfn;
    int x1, y1, x2, y2;
    vector<int> cls_label;
  } SEGITEMS;

  vector<SEGITEMS> lines_;
  int lines_id_;
  int label_dim_;
};

}  // namespace caffe

#endif /* SELECT_SEG_BINARY_TWO_FRAMES_HPP */

