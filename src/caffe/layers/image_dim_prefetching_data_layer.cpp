#include <string>
#include <vector>

#include "caffe/layers/image_dim_prefetching_data_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

  /* 
   * Jay add
   *
   */
/*
 notice:
 this code is based on the following implementation.
 https://bitbucket.org/deeplab/deeplab-public/
 */
template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  if (top.size() == 3) {
    output_data_dim_ = true;
  } else {
    output_data_dim_ = false;
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  if (output_data_dim_) {
    prefetch_data_dim_.mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  if (output_data_dim_) {
    caffe_copy(prefetch_data_dim_.count(), prefetch_data_dim_.cpu_data(),
               top[2]->mutable_cpu_data());
  }

  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
STUB_GPU_FORWARD(ImageDimPrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageDimPrefetchingData);

}  // namespace caffe
