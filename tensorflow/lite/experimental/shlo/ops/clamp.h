#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CLAMP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CLAMP_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct ClampOp {

 
  struct Attributes {};

  
 
};

ClampOp Create(ClampOp::Attributes);;
absl::Status Prepare(ClampOp& op, const Tensor& input, const Tensor& min_tensor,
                     const Tensor& max_tensor, Tensor& output);
absl::Status Evaluate(ClampOp& op, const Tensor& input, const Tensor& min_tensor,
                      const Tensor& max_tensor, Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CLAMP_H_
