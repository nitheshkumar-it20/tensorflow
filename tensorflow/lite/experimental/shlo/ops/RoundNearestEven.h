#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ROUND_NEAREST_EVEN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ROUND_NEAREST_EVEN_H_


#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct RoundNearestEvenOp {
  struct Attributes {};
};
RoundNearestEvenOp Create(RoundNearestEvenOp::Attributes);

absl::Status Prepare(RoundNearestEvenOp& op, const Tensor& input, Tensor& output);
absl::Status Evaluate(RoundNearestEvenOp& op, const Tensor& input, Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ROUND_NEAREST_EVEN_H_
