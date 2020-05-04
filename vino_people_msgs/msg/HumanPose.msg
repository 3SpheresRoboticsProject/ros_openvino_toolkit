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

vino_people_msgs/HumanPoseKeypoint[] keypoints # vector with the keypoints.

float32 score                       # score
sensor_msgs/RegionOfInterest roi 	# region of interest for a person's body

# The model human-pose-estimation-0001, available at:
# https://docs.openvinotoolkit.org/2019_R3.1/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html,
# provides the location of 18 joints, which are stored in the vector 
# `keypoints`. The keypoints are given by the following constants:
uint8 NOSE = 0
uint8 NECK = 1
uint8 RIGHT_SHOULDER = 2
uint8 RIGHT_ELBOW = 3
uint8 RIGHT_WRIST = 4
uint8 LEFT_SHOULDER = 5
uint8 LEFT_ELBOW = 6
uint8 LEFT_WRIST = 7
uint8 RIGHT_HIP = 8
uint8 RIGHT_KNEE = 9
uint8 RIGHT_ANKLE = 10
uint8 LEFT_HIP = 11
uint8 LEFT_KNEE = 12
uint8 LEFT_ANKLE = 13
uint8 RIGHT_EYE = 14
uint8 LEFT_EYE = 15
uint8 RIGHT_EAR = 16
uint8 LEFT_EAR = 17