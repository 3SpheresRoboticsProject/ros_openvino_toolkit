# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The model human-pose-estimation-0001, available at:
# https://docs.openvinotoolkit.org/2019_R3.1/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html,
# provides the location of 18 joints, which are stored in the vector 
# `keypoints`. The keypoints are:
# | Index | Keypoint Name  |
# | ----- | -------------- |
# |   0   | nose           |
# |   1   | neck           |
# |   2   | right shoulder |
# |   3   | right elbow    |
# |   4   | right wrist    |
# |   5   | left shoulder  |
# |   6   | left elbow     |
# |   7   | left wrist     |
# |   8   | right hip      |
# |   9   | right knee     |
# |  10   | right ankle    |
# |  11   | left hip       |
# |  12   | left knee      |
# |  13   | left ankle     |
# |  14   | right eye      |
# |  15   | left eye       |
# |  16   | right ear      |
# |  17   | left ear       |
vino_people_msgs/HumanPoseKeypoint[] keypoints

float32 score                       # score
sensor_msgs/RegionOfInterest roi 	# region of interest for a person's body