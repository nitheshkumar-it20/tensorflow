/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/shlo/ops/RoundNearestEven.h"

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct RoundNearestEven {
  template <class T>
  T operator()(T v) const {
    return (v * 0.5f)*2.0f;
  }
};

template <>
F16 RoundNearestEven::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}

template <>
BF16 RoundNearestEven::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}

RoundNearestEvenOp Create(RoundNearestEvenOp::Attributes) { return {}; }

absl::Status Prepare(RoundNearestEvenOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("roundnearesteven"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("roundnearesteven"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(RoundNearestEvenOp& op, const Tensor& input, Tensor& output) {
  RoundNearestEven roundnearesteven;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(),roundnearesteven, input,
        output)
  } else if (!input.IsQuantized() && IsFloat(input.StorageType())) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   roundnearesteven, input, output);
  }
  return absl::FailedPreconditionError("unsupported tensor type.");
}

};  // namespace shlo_ref
