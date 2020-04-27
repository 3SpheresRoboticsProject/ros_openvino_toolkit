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
 * @brief A header file with functions for interpreting peaks in the heatmap.
 * @file peak.h
 * 
 * Reference code provided by open_model_zoo at:
 * https://github.com/opencv/open_model_zoo/tree/master/demos/human_pose_estimation_demo
 */ 

#ifndef VINO_CORE_LIB_INFERENCES_HUMAN_POSE_ESTIMATION_PEAK_H
#define VINO_CORE_LIB_INFERENCES_HUMAN_POSE_ESTIMATION_PEAK_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "vino_core_lib/inferences/human_pose_estimation.h"

namespace human_pose_estimation
{

using Result = vino_core_lib::HumanPoseResult;

struct Peak
{
  Peak(const int id = -1,
       const cv::Point2f &pos = cv::Point2f(),
       const float score = 0.0f);

  int id;
  cv::Point2f pos;
  float score;
};

struct HumanPoseByPeaksIndices
{
  explicit HumanPoseByPeaksIndices(const int keypointsNumber);

  std::vector<int> peaksIndices;
  int nJoints;
  float score;
};

struct TwoJointsConnection
{
  TwoJointsConnection(const int firstJointIdx,
                      const int secondJointIdx,
                      const float score);

  int firstJointIdx;
  int secondJointIdx;
  float score;
};

void findPeaks(const std::vector<cv::Mat> &heatMaps,
               const float minPeaksDistance,
               std::vector<std::vector<Peak>> &allPeaks,
               int heatMapId);

std::vector<Result> groupPeaksToPoses(
    const std::vector<std::vector<Peak>> &allPeaks,
    const std::vector<cv::Mat> &pafs,
    const size_t keypointsNumber,
    const float midPointsScoreThreshold,
    const float foundMidPointsRatioThreshold,
    const int minJointsNumber,
    const float minSubsetScore);

class FindPeaksBody : public cv::ParallelLoopBody
{
public:
  FindPeaksBody(const std::vector<cv::Mat> &heatMaps, float minPeaksDistance,
                std::vector<std::vector<Peak>> &peaksFromHeatMap)
      : heatMaps(heatMaps),
        minPeaksDistance(minPeaksDistance),
        peaksFromHeatMap(peaksFromHeatMap) {}

  virtual void operator()(const cv::Range &range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
    }
  }

private:
  const std::vector<cv::Mat> &heatMaps;
  float minPeaksDistance;
  std::vector<std::vector<Peak>> &peaksFromHeatMap;
};

} // namespace human_pose_estimation

#endif // VINO_CORE_LIB_INFERENCES_HUMAN_POSE_ESTIMATION_PEAK_H
