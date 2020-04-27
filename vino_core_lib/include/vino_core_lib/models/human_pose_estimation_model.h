/*
 * Copyright (c) 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @brief A header file with declaration for HumanPoseEstimationModel Class
 * @file human_pose_estimation_model.h
 * 
 * Reference code provided by open_model_zoo at:
 * https://github.com/opencv/open_model_zoo/tree/master/demos/human_pose_estimation_demo
 * and
 * https://docs.openvinotoolkit.org/2019_R3.1/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html
 */

#ifndef VINO_CORE_LIB_MODELS_HUMAN_POSE_ESTIMATION_MODEL_H
#define VINO_CORE_LIB_MODELS_HUMAN_POSE_ESTIMATION_MODEL_H

#include <string>
#include "vino_core_lib/models/base_model.h"

namespace Models
{
/**
 * @class HumanPoseEstimationModel
 * @brief This class generates the human pose estimation model.
 */
class HumanPoseEstimationModel : public BaseModel
{
public:
    HumanPoseEstimationModel(const std::string&, int, int, int);

  /**
   * @brief Get the input name.
   * @return Input name.
   */
  inline const std::string getInputName() const
  {
    return input_;
  }
  /**
   * @brief Get the age from the detection reuslt.
   * @return Detected age.
   */
  inline const std::string getOutputKeypointsName() const
  {
    return output_keypoints_;
  }
  /**
   * @brief Get the gender from the detection reuslt.
   * @return Detected gender.
   */
  inline const std::string getOutputHeatmapName() const
  {
    return output_heatmap_;
  }
  /**
   * @brief Get the name of this detection model.
   * @return Name of the model.
   */
  const std::string getModelName() const override;

protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr&) override;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override;

 private:
  std::string input_;
  std::string output_keypoints_;
  std::string output_heatmap_;
};

}  // namespace Models

#endif  // VINO_CORE_LIB_MODELS_HUMAN_POSE_ESTIMATION_MODEL_H