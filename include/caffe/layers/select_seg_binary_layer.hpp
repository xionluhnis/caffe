#ifndef SELECT_SEG_BINARY_LAYER_HPP
#define SELECT_SEG_BINARY_LAYER_HPP

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    
/**
 * @brief Provides binary segmentation data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype, int NumImages>
class SelectSegBinaryLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SelectSegBinaryLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~SelectSegBinaryLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SelectSegBinary"; }
  virtual inline int ExactNumTopBlobs() const { return NumImages + 1; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void InternalThreadEntry();

 protected:
  Blob<Dtype> transformed_data_[NumImages];
  Blob<Dtype> class_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  struct SegItems {
    std::array<std::string, NumImages> imgfn;
    int x1, y1, x2, y2;
    vector<int> cls_label;
  };

  vector<SegItems> lines_;
  int lines_id_;
  int label_dim_;
  std::array<vector<int>, NumImages> img_dims_;
};

}  // namespace caffe

#endif /* SELECT_SEG_BINARY_LAYER_HPP */

