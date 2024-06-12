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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::ElementsAreArray;
using testing::NanSensitiveFloatEq;
using testing::Pointwise;

namespace shlo_ref {

template <>
struct ParamName<RoundNearestEvenOp> {
  static std::string Get() { return "roundnearesteven"; }
};

namespace {

struct  RoundNearestEven{
  template <class T>
  T operator()(T v) const {
    return (v * 0.5f)*2.0f;
  }
} roundnearesteven_ref;

template <>
F16 RoundNearestEven::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}

template <>
BF16 RoundNearestEven::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}

INSTANTIATE_TYPED_TEST_SUITE_P(RoundNearestEven, UnaryElementwiseOpShapePropagationTest,
                               RoundNearestEvenOp, TestParamNames);

INSTANTIATE_TYPED_TEST_SUITE_P(
    RoundNearestEven, UnaryElementwiseSameBaselineElementTypeConstraintTest,
    UnaryElementwiseConstraint1Types<RoundNearestEvenOp>, TestParamNames);

using UnsupportedTypes = WithOpTypes<
    RoundNearestEvenOp, ConcatTypes<BoolTestType, IntTestTypes, PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(RoundNearestEven, UnaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

template <class T>
struct FloatRoundNearestEvenTest : ::testing::Test {};

TYPED_TEST_SUITE(FloatRoundNearestEvenTest, FloatTestTypes, TestParamNames);

TYPED_TEST(FloatRoundNearestEvenTest, FloatTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const TensorType tensor_type =
      TensorType{.shape = shape, .element_type = TypeParam::kStorage};
  Tensor input_tensor{.type = tensor_type, .data = input_data.data()};
  Tensor output_tensor{.type = tensor_type, .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(input_data, expected_data.begin(), roundnearesteven_ref);

  auto op = Create(RoundNearestEvenOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(NanSensitiveFloatEq(), expected_data));
}

template <class T>
struct QuantizedRoundNearestEvenTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedRoundNearestEvenTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedRoundNearestEvenTest, PerTensorWorks) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const QuantizedPerTensorTensorType tensor_type = {
      .shape = shape,
      .element_type = QuantizedElementTypePerTensor(
          TypeParam::kStorage, zero_point, TypeParam::kExpressed, scale)};
  Tensor input_tensor{.type = tensor_type, .data = input_data.data()};
  Tensor output_tensor{.type = tensor_type, .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      input_data, expected_data.begin(), [zero_point, scale](auto v) {
        const ExpressedT dequantized_input = Dequantize(v, zero_point, scale);
        const ExpressedT dequantized_res = roundnearesteven_ref(dequantized_input);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(RoundNearestEvenOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

}  // namespace
}  // namespace shlo_ref