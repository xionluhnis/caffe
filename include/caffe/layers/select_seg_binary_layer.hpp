#ifndef SELECT_SEG_BINARY_LAYER_HPP
#define SELECT_SEG_BINARY_LAYER_HPP

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
template <typename Dtype>
class SelectSegBinaryLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SelectSegBinaryLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~SelectSegBinaryLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SelectSegBinary"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void InternalThreadEntry();

 protected:
  Blob<Dtype> transformed_label_;
  Blob<Dtype> class_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  typedef struct SegItems {
    std::string imgfn;
    std::string segfn;
    int x1, y1, x2, y2;
    vector<int> cls_label;
  } SEGITEMS;

  vector<SEGITEMS> lines_;
  int lines_id_;
  int label_dim_;
};

}  // namespace caffe

#endif /* SELECT_SEG_BINARY_LAYER_HPP */

