/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "general_post.h"

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "hiaiengine/log.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "tool_api.h"

using hiai::Engine;
using namespace std;

namespace {
// callback port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;
}
// namespace

// register custom data type
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

HIAI_StatusT GeneralPost::Init(
    const hiai::AIConfig &config,
    const vector<hiai::AIModelDescription> &model_desc) {
  // do noting
  return HIAI_OK;
}

bool GeneralPost::SendSentinel() {
  // can not discard when queue full
  HIAI_StatusT hiai_ret = HIAI_OK;
  shared_ptr<string> sentinel_msg(new (nothrow) string);
  do {
    hiai_ret = SendData(kSendDataPort, "string",
                        static_pointer_cast<void>(sentinel_msg));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL) {
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
}

void PostProcess(float *res, uint8_t *res_uint8, int32_t size,
    uint8_t model_type, uint32_t model_width, uint32_t model_height,
    uint32_t &output_width, uint32_t &output_height) {
  switch(model_type) {
    case 0: // SRCNN
      output_width = model_width;
      output_height = model_height;
      for (uint32_t i = 0; i < size; i++) {
        if (res[i] <= 0) {
          res_uint8[i] = 0;
        } else if (res[i] >= 1) {
          res_uint8[i] = 255;
        } else {
          res_uint8[i] = (uint8_t)(round(res[i] * 255));
        }
      }
      break;
    case 1: // FSRCNN
      output_width = model_width * 3;
      output_height = model_height * 3;
      for (uint32_t i = 0; i < size; i++) {
        if (res[i] <= 0) {
          res_uint8[i] = 0;
        } else if (res[i] >= 1) {
          res_uint8[i] = 255;
        } else {
          res_uint8[i] = (uint8_t)(round(res[i] * 255));
        }
      }
      break;
    case 3: //IDN  TBD
        output_width = model_width * 3;
        output_height = model_height * 3;
        for (uint32_t i = 0; i < size; i++) {
          if (res[i] <= 0) {
            res_uint8[i] = 0;
          } else if (res[i] >= 1) {
            res_uint8[i] = 255;
          } else {
            res_uint8[i] = (uint8_t)(round(res[i] * 255));
          }
        }
        break;
    case 2: // ESPCN
      output_width = model_width * 3;
      output_height = model_height * 3;
      uint32_t idx1 = 0;
      for (uint32_t c = 0; c < 9; c++) {
        for (uint32_t h = 0; h < model_height; h++) {
          for (uint32_t w = 0; w < model_width; w++) {
            // uint32_t idx2 = (h*3+c/3) * output_width + (w*3+c%3);
            uint32_t idx2 = (h*3+c%3) * output_width + (w*3+c/3);
            if (res[idx1] <= 0) {
              res_uint8[idx2] = 0;
            } else if (res[idx1] >= 1) {
              res_uint8[idx2] = 255;
            } else {
              res_uint8[idx2] = (uint8_t)(round(res[idx1] * 255));
            }
            idx1++;
          }
        }
      }
      break;
  }
}

void GenerateAndSaveImage(uint8_t *result, uint32_t height, uint32_t width,
    string file_path, uint8_t model_type, uint8_t is_colored) {
  // output file name
  int pos = file_path.find_last_of('/');
  string bicubic_name(file_path.substr(pos + 1));
  string output_name(file_path.substr(pos + 1));
  pos = bicubic_name.find_last_of('.');
  bicubic_name.insert(pos, "_bicubic");
  switch (model_type) {
    case 0 : output_name.insert(pos, "_srcnn"); break;
    case 1 : output_name.insert(pos, "_fsrcnn"); break;
    case 2: output_name.insert(pos, "_espcn"); break;
    case 3 : output_name.insert(pos,"_idn");  break;
  }

  cv::Mat mat_out_y(height, width, CV_8U, result);

  // generate colored image
  if (is_colored) {
    // read BGR image
    cv::Mat mat = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);

    // bicubic
    cv::Mat mat_bicubic; 
    cv::resize(mat, mat_bicubic, cv::Size(0, 0), 3, 3, cv::INTER_CUBIC);
    cv::imwrite(bicubic_name, mat_bicubic);

    // BGR2YCrCb and bicubic
    cv::Mat mat_ycrcb, mat_out_ycrcb; 
    cv::cvtColor(mat, mat_ycrcb, cv::COLOR_BGR2YCrCb);
    cv::resize(mat_ycrcb, mat_out_ycrcb, cv::Size(0, 0), 3, 3, cv::INTER_CUBIC);

    // replace Y
    vector<cv::Mat> channels;
    cv::split(mat_out_ycrcb, channels);
    channels[0] = mat_out_y;
    cv::merge(channels, mat_out_ycrcb);

    // YCrCb2BGR
    cv::Mat mat_out_bgr; 
    cv::cvtColor(mat_out_ycrcb, mat_out_bgr, cv::COLOR_YCrCb2BGR);
    cv::imwrite(output_name, mat_out_bgr);
  } else {
    // Gray
    cv::Mat mat_gray, mat_bicubic; 
    mat_gray = cv::imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::resize(mat_gray, mat_bicubic, cv::Size(0, 0), 3, 3, cv::INTER_CUBIC);
    cv::imwrite(bicubic_name, mat_bicubic);
    cv::imwrite(output_name, mat_out_y);
  }
}

HIAI_StatusT GeneralPost::SuperResolutionPostProcess(
    const std::shared_ptr<EngineTrans> &result) {
  string file_path = result->image_info.path;
  // check vector
  if (result->inference_res.empty()) {
    ERROR_LOG("Failed to deal file=%s. Reason: inference result empty.",
              file_path.c_str());
    return HIAI_ERROR;
  }

  // only need to get first one
  Output out = result->inference_res[0];
  int32_t size = out.size / sizeof(float);
  if (size <= 0) {
    ERROR_LOG("Failed to deal file=%s. Reason: inference result size=%d error.",
              file_path.c_str(), size);
    return HIAI_ERROR;
  }

  // transform results
  float *res = new (nothrow) float[size];
  if (res == nullptr) {
    ERROR_LOG("Failed to deal file=%s. Reason: new float array failed.",
              file_path.c_str());
    return HIAI_ERROR;
  }
  errno_t mem_ret = memcpy_s(res, sizeof(float) * size, out.data.get(),
                             out.size);
  if (mem_ret != EOK) {
    delete[] res;
    ERROR_LOG("Failed to deal file=%s. Reason: call memcpy_s failed.",
              file_path.c_str());
    return HIAI_ERROR;
  }

  uint8_t res_uint8[size];
  uint32_t output_width, output_height;

  // post process
  PostProcess(res, res_uint8, size, result->console_params.model_type,
      result->console_params.model_width, result->console_params.model_height,
      output_width, output_height);

  // generate and save BGR image
  GenerateAndSaveImage(res_uint8, output_height, output_width, file_path,
      result->console_params.model_type, result->console_params.is_colored);

  delete[] res;
  INFO_LOG("Success to deal file=%s.", file_path.c_str());
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("general_post", GeneralPost, INPUT_SIZE) {
  HIAI_StatusT ret = HIAI_OK;

  // check arg0
  if (arg0 == nullptr) {
    ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
    return HIAI_ERROR;
  }

  // just send to callback function when finished
  shared_ptr<EngineTrans> result = static_pointer_cast<EngineTrans>(arg0);
  if (result->is_finished) {
    if (SendSentinel()) {
      return HIAI_OK;
    }
    ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
    ERROR_LOG("Please stop this process manually.");
    return HIAI_ERROR;
  }

  // inference failed
  if (result->err_msg.error) {
    ERROR_LOG("%s", result->err_msg.err_msg.c_str());
    return HIAI_ERROR;
  }

  // arrange result
  return SuperResolutionPostProcess(result);
}
