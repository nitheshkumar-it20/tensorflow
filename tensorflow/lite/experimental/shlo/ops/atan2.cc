#include "tensorflow/lite/experimental/shlo/ops/atan2.h"

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Atan2 {
  template <class T>
  T operator()(T x, T y) const {
    return std::atan2(y, x);
  }
};

template <>
F16 Atan2::operator()<F16>(F16 x, F16 y) const {
  return F16(operator()(static_cast<float>(x), static_cast<float>(y)));
}

template <>
BF16 Atan2::operator()<BF16>(BF16 x, BF16 y) const {
  return BF16(operator()(static_cast<float>(x), static_cast<float>(y)));
}

Atan2Op Create(Atan2Op::Attributes) { return {}; }

absl::Status Prepare(Atan2Op& op, const Tensor& input_x, const Tensor& input_y, Tensor& output) {
  // Check if input tensors have the same shape
 /*if (input_x.shape() != input_y.shape()) {
    return absl::InvalidArgumentError("Input tensors must have the same shape.");
  }*/

  // Propagate shape from input_x to output
  SHLO_REF_RETURN_ON_ERROR(Propagate(input_x.shape(), input_y.shape(),output.shape()));

  // Check supported types for both input_x and input_y
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("atan"), input_x, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("atan"), input_y, IsFloatTensor, IsQuantizedPerTensorTensor));

  // Check if the baseline types of all tensors are the same
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("atan"), input_x, output));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("atan"), input_y, output));

  return absl::OkStatus();
}


 

absl::Status Evaluate(Atan2Op& op, const Tensor& input_x, const Tensor& input_y, Tensor& output) {
  Atan2 atan2;
  if (input_x.IsPerTensorQuantized() && input_y.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input_x.quantized_per_tensor_element_type().StorageType(),
        input_x.quantized_per_tensor_element_type().ExpressedType(), atan2, input_x, input_y, output)
  } else if (!input_x.IsQuantized() && !input_y.IsQuantized() && IsFloat(input_x.StorageType()) && IsFloat(input_y.StorageType())) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input_x.tensor_element_type(),
                   atan2, input_x, input_y, output);
  } else {
    return absl::FailedPreconditionError("Unsupported tensor type.");
  }
  return absl::OkStatus();
}

}  // namespace shlo_ref
