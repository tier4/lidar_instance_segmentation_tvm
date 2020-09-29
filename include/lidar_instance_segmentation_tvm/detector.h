/*
 * Copyright 2020 TierIV. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// tvm
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "cluster2d.h"
#include "feature_generator.h"
#include "lidar_instance_segmentation_tvm/node.h"

#include <pcl_ros/transforms.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <memory>

class LidarApolloInstanceSegmentation : public LidarInstanceSegmentationInterface {
 public:
  LidarApolloInstanceSegmentation();
  ~LidarApolloInstanceSegmentation(){};
  bool detectDynamicObjects(const sensor_msgs::PointCloud2& input,
                            autoware_perception_msgs::DynamicObjectWithFeatureArray& output) override;

  std::shared_ptr<float> getInferredData() override;
  int getRange() override;
  int getWidth() override;
  int getHeight() override;
  std::vector<int> getCluster2dIdImage() override;

 private:
  bool transformCloud(const sensor_msgs::PointCloud2& input, sensor_msgs::PointCloud2& transformed_cloud,
                      float z_offset);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  std::shared_ptr<Cluster2D> cluster2d_;
  std::shared_ptr<FeatureGenerator> feature_generator_;
  float score_threshold_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string target_frame_;
  float z_offset_;

  int loop_count_ = 0;
  int num_save_feature_;

  std::shared_ptr<float> inferred_data_;
  int range_;
  int width_;
  int height_;

  // tvm
  int in_ndim_ = 4;
  int nbytes_float32_ = 4;
  int dtype_code_ = kDLFloat;
  int dtype_bits_ = 32;
  int dtype_lanes_ = 1;
  int device_cpu_ = kDLCPU;
  int device_gpu_ = kDLGPU;
  int device_cpu_id_ = 0;
  int device_gpu_id_ = 0;
  int out_ndim_ = 4;
  int out_channels = 12;
  void* handle_;
};
