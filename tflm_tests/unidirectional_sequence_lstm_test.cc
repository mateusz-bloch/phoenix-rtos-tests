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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include <cstring>

namespace tflite {
namespace testing {

namespace {
// LSTM internal setting (e.g., nonlinear activation type)
// Only UnidirectionalLSTM is supported now
constexpr TfLiteUnidirectionalSequenceLSTMParams kDefaultBuiltinData = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/6,
    /*.proj_clip=*/3,
    /*.time_major=*/false,
    /*.asymmetric_quantize_inputs=*/true,
    /*diagonal_recurrent_tensors=*/false};
}  // namespace

GateOutputCheckData<4, 4> Get2X2GateOutputCheckData() {
  GateOutputCheckData<4, 4> gate_data;
  const float input_data[4] = {
      0.2, 0.3,    // batch1
      -0.98, 0.62  // batch2
  };
  std::memcpy(gate_data.input_data, input_data, 4 * sizeof(float));

  const float hidden_state[4] = {
      -0.1, 0.2,  // batch1
      -0.3, 0.5   // batch2
  };
  std::memcpy(gate_data.hidden_state, hidden_state, 4 * sizeof(float));

  const float cell_state[4] = {
      -1.3, 6.2,  // batch1
      -7.3, 3.5   // batch2
  };
  std::memcpy(gate_data.cell_state, cell_state, 4 * sizeof(float));

  // Use the forget gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[-10,-10],[-20,-20]][0.2,
  // +[[-10,-10],[-20,-20]][-0.1, 0.2]+[1,2]) = sigmoid([-5,-10]) =
  // [6.69285092e-03, 4.53978687e-05] (Batch1)
  // Similarly, we have [0.93086158 0.9945137 ] for batch 2
  const float expected_forget_gate_output[4] = {6.69285092e-3f, 4.53978687e-5f,
                                                0.93086158, 0.9945137};
  std::memcpy(gate_data.expected_forget_gate_output,
              expected_forget_gate_output, 4 * sizeof(float));

  // Use the input gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[10,10],[20,20]][0.2, 0.3]
  // +[[10,10],[20,20]][-0.1, 0.2]+[-1,-2]) = sigmoid([5,10]) =
  // [0.99330715, 0.9999546]
  // Similarly, we have [0.06913842 0.0054863 ] for batch 2
  const float expected_input_gate_output[4] = {0.99330715, 0.9999546,
                                               0.06913842, 0.0054863};
  std::memcpy(gate_data.expected_input_gate_output, expected_input_gate_output,
              4 * sizeof(float));

  // Use the output gate parameters to test normnal gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = sigmoid([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  // Similarly, we have [[0.46008512 0.46008512]] for batch 2
  const float expected_output_gate_output[4] = {
      0.6456563062257954, 0.6456563062257954, 0.46008512, 0.46008512};
  std::memcpy(gate_data.expected_output_gate_output,
              expected_output_gate_output, 4 * sizeof(float));

  // Use the cell(modulation) gate parameters to tanh output
  // output = tanh(W_i*i+W_h*h+b) = tanh([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = tanh([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  // Similarly, we have [-0.1586485 -0.1586485] for batch 2
  const float expected_cell_gate_output[4] = {
      0.5370495669980353, 0.5370495669980353, -0.1586485, -0.1586485};
  std::memcpy(gate_data.expected_cell_gate_output, expected_cell_gate_output,
              4 * sizeof(float));

  // Cell = forget_gate*cell + input_gate*cell_gate
  // Note -6.80625824 is clipped to -6
  const float expected_updated_cell[4] = {0.52475447, 0.53730665, -6,
                                          3.47992756};
  std::memcpy(gate_data.expected_updated_cell, expected_updated_cell,
              4 * sizeof(float));

  // Use the updated cell state to update the hidden state
  // tanh(expected_updated_cell) * expected_output_gate_output
  const float expected_updated_hidden[4] = {0.31079388, 0.3169827, -0.46007947,
                                            0.45921249};
  std::memcpy(gate_data.expected_updated_hidden, expected_updated_hidden,
              4 * sizeof(float));
  return gate_data;
}

// TODO(b/253466487): document how the golden values are arrived at
LstmEvalCheckData<12, 4, 12> Get2X2LstmEvalCheckData() {
  LstmEvalCheckData<12, 4, 12> eval_data;
  const float input_data[12] = {
      0.2,   0.3,  0.2,  0.3,  0.2,  0.3,   // batch one
      -0.98, 0.62, 0.01, 0.99, 0.49, -0.32  // batch two
  };
  std::memcpy(eval_data.input_data, input_data, 12 * sizeof(float));

  // Initialize hidden state as zeros
  const float hidden_state[4] = {};
  std::memcpy(eval_data.hidden_state, hidden_state, 4 * sizeof(float));

  // The expected model output after 3 time steps using the fixed input and
  // parameters
  const float expected_output[12] = {
      0.26455893,      0.26870455,      0.47935803,
      0.47937014,      0.58013272,      0.58013278,  // batch1
      -1.41184672e-3f, -1.43329117e-5f, 0.46887168,
      0.46891281,      0.50054074,      0.50054148  // batch2
  };
  std::memcpy(eval_data.expected_output, expected_output, 12 * sizeof(float));

  const float expected_hidden_state[4] = {
      0.58013272, 0.58013278,  // batch1
      0.50054074, 0.50054148   // batch2
  };
  std::memcpy(eval_data.expected_hidden_state, expected_hidden_state,
              4 * sizeof(float));

  const float expected_cell_state[4] = {
      0.89740515, 0.8974053,  // batch1
      0.80327607, 0.80327785  // batch2
  };
  std::memcpy(eval_data.expected_cell_state, expected_cell_state,
              4 * sizeof(float));
  return eval_data;
}

LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
Create2x3x2X2FloatNodeContents(const float* input_data,
                               const float* hidden_state_data,
                               const float* cell_state_data) {
  // Parameters for different gates
  // negative large weights for forget gate to make it really forget
  const GateData<float, float, 2, 2> forget_gate_data = {
      /*.activation_weight=*/{-10, -10, -20, -20},
      /*.recurrent_weight=*/{-10, -10, -20, -20},
      /*.fused_bias=*/{1, 2},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};
  // positive large weights for input gate to make it really remember
  const GateData<float, float, 2, 2> input_gate_data = {
      /*.activation_weight=*/{10, 10, 20, 20},
      /*.recurrent_weight=*/{10, 10, 20, 20},
      /*.fused_bias=*/{-1, -2},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};
  // all ones to test the behavior of tanh at normal range (-1,1)
  const GateData<float, float, 2, 2> cell_gate_data = {
      /*.activation_weight=*/{1, 1, 1, 1},
      /*.recurrent_weight=*/{1, 1, 1, 1},
      /*.fused_bias=*/{0, 0},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};
  // all ones to test the behavior of sigmoid at normal range (-1. 1)
  const GateData<float, float, 2, 2> output_gate_data = {
      /*.activation_weight=*/{1, 1, 1, 1},
      /*.recurrent_weight=*/{1, 1, 1, 1},
      /*.fused_bias=*/{0, 0},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};

  LstmNodeContent<float, float, float, float, 2, 3, 2, 2> float_node_contents(
      kDefaultBuiltinData, forget_gate_data, input_gate_data, cell_gate_data,
      output_gate_data);

  if (input_data != nullptr) {
    float_node_contents.SetInputData(input_data);
  }
  if (hidden_state_data != nullptr) {
    float_node_contents.SetHiddenStateData(hidden_state_data);
  }
  if (cell_state_data != nullptr) {
    float_node_contents.SetCellStateData(cell_state_data);
  }
  return float_node_contents;
}

NodeQuantizationParameters Get2X2Int8LstmQuantizationSettings() {
  NodeQuantizationParameters quantization_settings;
  quantization_settings.activation_type = kTfLiteInt8;
  quantization_settings.weight_type = kTfLiteInt8;
  quantization_settings.cell_type = kTfLiteInt16;
  quantization_settings.bias_type = kTfLiteInt32;
  quantization_settings.nonlinear_activation_input_scale =
      0.00024414062;  // std::pow(2.0f, -12.0f)
  quantization_settings.nonlinear_activation_output_scale =
      0.00003051757;  // std::pow(2.0f, -15.0f)

  // state quantization parameters
  quantization_settings.input = {/*scale=*/0.00784313725490196, /*zp=*/0,
                                 /*symmetry=*/false};
  quantization_settings.output = {/*scale=*/0.004705882165580988, /*zp=*/-21,
                                  /*symmetry=*/false};
  quantization_settings.hidden_state = {/*scale=*/0.004705882165580988,
                                        /*zp=*/-21, /*symmetry=*/false};
  quantization_settings.cell_state = {/*scale=*/0.00024414062, /*zp=*/0,
                                      /*symmetry=*/true};

  // gate quantization parameters
  quantization_settings.forget_gate = {
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.0012351397251814111, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.input_gate = {
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.0012351397251814111, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.cell_gate = {
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/6.175698625907056e-5, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.output_gate = {
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/6.175698625907056e-5, /*zp=*/0, /*symmetry=*/true}};
  return quantization_settings;
}

NodeQuantizationParameters Get2X2Int16LstmQuantizationSettings() {
  NodeQuantizationParameters quantization_settings;
  quantization_settings.activation_type = kTfLiteInt16;
  quantization_settings.weight_type = kTfLiteInt8;
  quantization_settings.cell_type = kTfLiteInt16;
  quantization_settings.bias_type = kTfLiteInt64;
  quantization_settings.nonlinear_activation_input_scale =
      0.00024414062;  // std::pow(2.0f, -12.0f)
  quantization_settings.nonlinear_activation_output_scale =
      0.00003051757;  // std::pow(2.0f, -15.0f)

  // state quantization parameters
  quantization_settings.input = {/*scale=*/3.0518044e-5, /*zp=*/0,
                                 /*symmetry=*/true};
  quantization_settings.output = {/*scale=*/2.1362956633198035e-05, /*zp=*/0,
                                  /*symmetry=*/true};
  quantization_settings.hidden_state = {/*scale=*/2.1362956633198035e-05,
                                        /*zp=*/0,
                                        /*symmetry=*/true};
  quantization_settings.cell_state = {/*scale=*/0.00024414062, /*zp=*/0,
                                      /*symmetry=*/true};

  // gate quantization parameters
  quantization_settings.forget_gate = {
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/4.8059911474468205e-06, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.input_gate = {
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/4.8059911474468205e-06, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.cell_gate = {
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/2.40299557372341e-07, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.output_gate = {
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/2.40299557372341e-07, /*zp=*/0, /*symmetry=*/true}};
  return quantization_settings;
}

LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
Create2x3x2X2Int8NodeContents(const float* input_data,
                              const float* hidden_state,
                              const float* cell_state) {
  auto float_node_content =
      Create2x3x2X2FloatNodeContents(input_data, hidden_state, cell_state);
  const auto quantization_settings = Get2X2Int8LstmQuantizationSettings();
  return CreateIntegerNodeContents<int8_t, int8_t, int32_t, int16_t, 2, 3, 2,
                                   2>(quantization_settings,
                                      /*fold_zero_point=*/true,
                                      float_node_content);
}

LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2, 2>
Create2x3x2X2Int16NodeContents(const float* input_data,
                               const float* hidden_state,
                               const float* cell_state) {
  auto float_node_content =
      Create2x3x2X2FloatNodeContents(input_data, hidden_state, cell_state);
  const auto quantization_settings = Get2X2Int16LstmQuantizationSettings();
  return CreateIntegerNodeContents<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>(quantization_settings,
                                      /*fold_zero_point=*/false,
                                      float_node_content);
}

}  // namespace testing
}  // namespace tflite

namespace tflite {
namespace testing {
namespace {

constexpr int kLstmMaxNumInputOutputTensors = 24 + 1;

// Validate the output result array with golden values
template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUnidirectionalLSTMInteger(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(
      registration, node_contents.GetTensors(), kLstmMaxNumInputOutputTensors,
      node_contents.KernelInputs(), node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const auto& quantization_settings = node_contents.QuantizationSettings();

// CMSIS-NN does not use the hidden state and cell state tensors so these tests
// fail.
#if !defined(CMSIS_NN)
  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetHiddenStateData(), batch_size * state_dimension,
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point,
             dequantized_hidden_state);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetCellStateData(), batch_size * state_dimension,
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point,
             dequantized_cell_state);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);
#endif

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(node_contents.GetOutputData(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output.scale,
             quantization_settings.output.zero_point, dequantized_output);
  ValidateResultGoldens(eval_check_data.expected_output, dequantized_output,
                        batch_size * state_dimension, hidden_state_tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestUnidirectionalLSTMFloat(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents) {
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(
      registration, node_contents.GetTensors(), kLstmMaxNumInputOutputTensors,
      node_contents.KernelInputs(), node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        node_contents.GetHiddenStateData(),
                        batch_size * state_dimension, hidden_state_tolerance);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        node_contents.GetCellStateData(),
                        batch_size * state_dimension, cell_state_tolerance);
  ValidateResultGoldens(eval_check_data.expected_output,
                        node_contents.GetOutputData(),
                        batch_size * state_dimension, hidden_state_tolerance);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
TF_LITE_MICRO_TEST(TestUnidirectionalLSTMFloat) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float tolerance = 1e-6;
  tflite::testing::TestUnidirectionalLSTMFloat(kernel_eval_data, tolerance,
                                               tolerance, float_node_contents);
}

TF_LITE_MICRO_TEST(TestUnidirectionalLSTMInt8) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTMInteger(
      kernel_eval_data, hidden_state_tolerance, cell_state_tolerance,
      int8_node_contents);
}

TF_LITE_MICRO_TEST(TestUnidirectionalLSTMInt16) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-3;  // actually very close to 1e-4
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTMInteger(
      kernel_eval_data, hidden_state_tolerance, cell_state_tolerance,
      int16_node_contents);
}
TF_LITE_MICRO_TESTS_END
