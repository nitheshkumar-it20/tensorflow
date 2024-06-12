#include "clamp.h"

#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref {


namespace {



template <class T>
struct ClampTest : ::testing::Test {};

TYPED_TEST_SUITE(ClampTest, IntTestTypes, TestParamNames);

TYPED_TEST(ClampTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({3});
  
  Vector<StorageT> operand_data{2, 3, -1};

  Vector<StorageT> min_data{1, 5, -5};

   Vector<StorageT> max_data{3, 7, -3};
 
  Vector<StorageT> output_data(shape.NumElements());

  Tensor operand{.type = TensorType{.shape = shape,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(ClampOp::Attributes{
     
  });

  Vector<StorageT> expected_data{2, 5, -3};
  
  ASSERT_OK(Prepare(op, operand_data, min_data,max_data,output_data));
  ASSERT_OK(Evaluate(op, operand_data, min_data,max_data,output_data));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}
}  // namespace
}  // namespace shlo_ref
