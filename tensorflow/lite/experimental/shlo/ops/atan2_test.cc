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

#include "tensorflow/lite/experimental/shlo/ops/atan2.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
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
struct ParamName<Atan2Op> {
  static std::string Get() { return "atan"; }
};

namespace {

struct Atan2 {
  template <class T>
  T operator()(T x, T y) const {
    return std::atan2(y,x);
  }
} atan2_ref;

template <>
F16 Atan2::operator()<F16>(F16 x, F16 y) const {
  return F16(operator()(static_cast<float>(x), static_cast<float>(y)));
}

template <>
BF16 Atan2::operator()<BF16>(BF16 x, BF16 y) const {
  return BF16(operator()(static_cast<float>(x), static_cast<float>(y)));
}

INSTANTIATE_TYPED_TEST_SUITE_P(Atan2, BinaryElementwiseOpShapePropagationTest,
                               Atan2Op, TestParamNames);

using Atan2BaselineConstraintTypes = BinaryElementwiseBaselineConstraintTypes<
    Atan2Op, ConcatTypes<BaselineConstraintFloatTypes,
                         BaselineConstraintQuantizedPerTensorTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    Atan2, BinaryElementwiseSameBaselineElementTypeConstraintTest,
    Atan2BaselineConstraintTypes, TestParamNames);

using UnsupportedTypes = WithOpTypes<Atan2Op, PerAxisQuantizedTestTypes>;

INSTANTIATE_TYPED_TEST_SUITE_P(Atan2, BinaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

template <class T>
struct FloatAtan2Test : ::testing::Test {};

TYPED_TEST_SUITE(FloatAtan2Test, FloatTestTypes, TestParamNames);

TYPED_TEST(FloatAtan2Test, FloatTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data_x = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> input_data_y = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const TensorType tensor_type =
      TensorType{.shape = shape, .element_type = TypeParam::kStorage};
  Tensor input_tensor_x{.type = tensor_type, .data = input_data_x.data()};
  Tensor input_tensor_y{.type = tensor_type, .data = input_data_y.data()};
  Tensor output_tensor{.type = tensor_type, .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  std::transform(input_data_x.begin(), input_data_x.end(), input_data_y.begin(),
                 expected_data.begin(),
                 [](float x, float y) { return atan2_ref(x, y); });

  auto op = Create(Atan2Op::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor_x, input_tensor_y, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor_x, input_tensor_y, output_tensor));
  EXPECT_THAT(output_data, Pointwise(NanSensitiveFloatEq(), expected_data));
}


template <class T>
struct QuantizedAtan2Test : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedAtan2Test, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedAtan2Test, PerTensorWorks) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
 Vector<StorageT> input_data_x = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> input_data_y = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const QuantizedPerTensorTensorType tensor_type = {
      .shape = shape,
      .element_type = QuantizedElementTypePerTensor(
          TypeParam::kStorage, zero_point, TypeParam::kExpressed, scale)};
   Tensor input_tensor_x{.type = tensor_type, .data = input_data_x.data()};
  Tensor input_tensor_y{.type = tensor_type, .data = input_data_y.data()};
  Tensor output_tensor{.type = tensor_type, .data = output_data.data()};


  Vector<StorageT> expected_data(shape.NumElements());
  std::transform(
      input_data_x.begin(), input_data_x.end(), input_data_y.begin(),
expected_data.begin(), [zero_point, scale](auto x, auto y) { const ExpressedT
dequantized_input_x = Dequantize(x, zero_point, scale); const ExpressedT
dequantized_input_y = Dequantize(y, zero_point, scale); const ExpressedT
dequantized_res = atan2_ref(dequantized_input_x, dequantized_input_y); return
Quantize<TypeParam::kStorage, TypeParam::kExpressed>( dequantized_res,
zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(Atan2Op::Attributes{});
    ASSERT_OK(Prepare(op, input_tensor_x, input_tensor_y, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor_x, input_tensor_y, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}


}  // namespace
}  // namespace shlo_ref
