#include "clamp.h"
#include <algorithm>

#include "tensorflow/lite/experimental/shlo/tensor.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"

namespace shlo_ref {
ClampOp Create(ClampOp::Attributes attributes) {
  return {};
}

absl::Status Prepare(ClampOp& op, const Tensor& input, const Tensor& min_tensor,
                     const Tensor& max_tensor, Tensor& output) {
  // Check if input and output shapes match
  if (input.shape()!=output.shape()) {
    return absl::InvalidArgumentError("Input and output shapes must match.");
  }
  // Check if input and min tensor shapes match
  if (input.shape()!=min_tensor.shape()) {
    return absl::InvalidArgumentError("Input and min tensor shapes must match.");
  }
  // Check if input and max tensor shapes match
  if (input.shape()!=max_tensor.shape()) {
    return absl::InvalidArgumentError("Input and max tensor shapes must match.");
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status ClampTensor(const Tensor& input, const Tensor& min_tensor, const Tensor& max_tensor,
                 Tensor& output) {
                  using StorageT=StorageType<storage_type>;
  // const T min_val = *reinterpret_cast<const T*>(min_tensor.data);
  // const T max_val = *reinterpret_cast<const T*>(max_tensor.data);
  const StorageT* min_buffer = min_tensor.GetDataAs<storage_type>();
  const StorageT* max_buffer = max_tensor.GetDataAs<storage_type>();
  const StorageT* input_buffer = input.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize num_elements = input.NumElements();
  // const T* input_data = reinterpret_cast<const T*>(input.data);
  // T* output_data = reinterpret_cast<T*>(output.data);
  for (DimensionSize i = 0; i < num_elements; ++i) {
    output_buffer[i] = std::min(std::max(input_buffer[i], min_buffer[i]), max_buffer[i]);
  }
   return absl::OkStatus();
}

absl::Status Evaluate(ClampOp& op, const Tensor& input, const Tensor& min_tensor,
                      const Tensor& max_tensor, Tensor& output) {
DISPATCH_INT_FLOAT( ClampTensor,output.StorageType(),input, min_tensor,max_tensor,   output);
  return absl::FailedPreconditionError(
      "stablehlo.transpose: Unsupported tensor type.");

}  // namespace shlo_ref
}