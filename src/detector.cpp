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

#include "lidar_instance_segmentation_tvm/detector.h"
#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <boost/filesystem.hpp>
#include "lidar_instance_segmentation_tvm/feature_map.h"

#include <ros/package.h>

#include <stdio.h>
#include <fstream>

LidarApolloInstanceSegmentation::LidarApolloInstanceSegmentation() : nh_(""), pnh_("~"), tf_listener_(tf_buffer_) {
  bool use_intensity_feature, use_constant_feature;
  std::string model_file_path;
  std::string model_json_path;
  std::string model_params_path;
  pnh_.param<float>("score_threshold", score_threshold_, 0.8);
  pnh_.param<int>("range", range_, 60);
  pnh_.param<int>("width", width_, 640);
  pnh_.param<int>("height", height_, 640);
  pnh_.param<std::string>("model_file_path", model_file_path, "data/bcnn.so");
  pnh_.param<std::string>("model_json_path", model_json_path, "data/model_graph.json");
  pnh_.param<std::string>("model_params_path", model_params_path, "data/model_graph.params");
  pnh_.param<bool>("use_intensity_feature", use_intensity_feature, true);
  pnh_.param<bool>("use_constant_feature", use_constant_feature, true);
  pnh_.param<std::string>("target_frame", target_frame_, "base_link");
  pnh_.param<float>("z_offset", z_offset_, 2);
  pnh_.param<int>("num_save_feature", num_save_feature_, 0);
  // in_shape_ = {1, 6, 672, 672};

  // load tvm weight file
  std::cout << model_file_path << std::endl;
  tvm::runtime::Module bcnn_lib_ = tvm::runtime::Module::LoadFromFile(model_file_path);
  std::ifstream json_in(model_json_path, std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  std::ifstream params_in(model_params_path, std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  tvm::runtime::Module mod =
      (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, bcnn_lib_, device_gpu_, device_gpu_id_);
  this->handle_ = new tvm::runtime::Module(mod);

  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  // feature map generator: pre process
  feature_generator_ =
      std::make_shared<FeatureGenerator>(width_, height_, range_, use_intensity_feature, use_constant_feature);

  // cluster: post process
  cluster2d_ = std::make_shared<Cluster2D>(width_, height_, range_);
}

bool LidarApolloInstanceSegmentation::transformCloud(const sensor_msgs::PointCloud2& input,
                                                     sensor_msgs::PointCloud2& transformed_cloud, float z_offset) {
  // transform pointcloud to tagret_frame
  if (target_frame_ != input.header.frame_id) {
    try {
      geometry_msgs::TransformStamped transform_stamped;
      transform_stamped =
          tf_buffer_.lookupTransform(target_frame_, input.header.frame_id, input.header.stamp, ros::Duration(0.5));
      Eigen::Matrix4f affine_matrix = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
      pcl_ros::transformPointCloud(affine_matrix, input, transformed_cloud);
      transformed_cloud.header.frame_id = target_frame_;
    } catch (tf2::TransformException& ex) {
      ROS_WARN("%s", ex.what());
      return false;
    }
  } else {
    transformed_cloud = input;
  }

  // move pointcloud z_offset in z axis
  sensor_msgs::PointCloud2 pointcloud_with_z_offset;
  Eigen::Affine3f z_up_translation(Eigen::Translation3f(0, 0, z_offset));
  Eigen::Matrix4f z_up_transform = z_up_translation.matrix();
  pcl_ros::transformPointCloud(z_up_transform, transformed_cloud, transformed_cloud);

  return true;
}

bool LidarApolloInstanceSegmentation::detectDynamicObjects(
    const sensor_msgs::PointCloud2& input, autoware_perception_msgs::DynamicObjectWithFeatureArray& output) {
  // move up pointcloud z_offset in z axis
  sensor_msgs::PointCloud2 transformed_cloud;
  transformCloud(input, transformed_cloud, z_offset_);

  // convert from ros to pcl
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pointcloud_raw_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(transformed_cloud, *pcl_pointcloud_raw_ptr);

  // generate feature map
  std::shared_ptr<FeatureMapInterface> feature_map_ptr = feature_generator_->generate(pcl_pointcloud_raw_ptr);

  // tvm inference
  DLTensor* x;
  DLTensor* y;

  int64_t in_shape[4] = {1, feature_map_ptr->channels, height_, width_};
  TVMArrayAlloc(in_shape, in_ndim_, dtype_code_, dtype_bits_, dtype_lanes_, device_cpu_, device_cpu_id_, &x);

  memcpy(x->data, feature_map_ptr->map_data.data(), width_ * height_ * feature_map_ptr->channels * sizeof(float));
  tvm::runtime::Module* mod = (tvm::runtime::Module*)handle_;
  tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");

  set_input("input", x);
  tvm::runtime::PackedFunc run = mod->GetFunction("run");
  run();

  int64_t out_shape_[4] = {1, out_channels, height_, width_};
  TVMArrayAlloc(out_shape_, out_ndim_, dtype_code_, dtype_bits_, dtype_lanes_, device_cpu_, device_cpu_id_, &y);
  tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");

  get_output(0, y);

  std::shared_ptr<float> inferred_data(new float[width_ * height_ * out_channels], std::default_delete<float[]>());
  memcpy(inferred_data.get(), static_cast<float*>(y->data), width_ * height_ * out_channels * sizeof(float));

  // post process
  const float objectness_thresh = 0.5;
  pcl::PointIndices valid_idx;
  valid_idx.indices.resize(pcl_pointcloud_raw_ptr->size());
  std::iota(valid_idx.indices.begin(), valid_idx.indices.end(), 0);
  cluster2d_->cluster(inferred_data, pcl_pointcloud_raw_ptr, valid_idx, objectness_thresh,
                      true /*use all grids for clustering*/);
  const float height_thresh = 0.5;
  const int min_pts_num = 3;
  cluster2d_->getObjects(score_threshold_, height_thresh, min_pts_num, output, input.header);

  // move down pointcloud z_offset in z axis
  for (int i = 0; i < output.feature_objects.size(); i++) {
    sensor_msgs::PointCloud2 transformed_cloud;
    transformCloud(output.feature_objects.at(i).feature.cluster, transformed_cloud, -z_offset_);
    output.feature_objects.at(i).feature.cluster = transformed_cloud;
  }

  output.header = input.header;
  inferred_data_ = inferred_data;

  TVMArrayFree(x);
  TVMArrayFree(y);

  return true;
}

std::shared_ptr<float> LidarApolloInstanceSegmentation::getInferredData() { return inferred_data_; }

int LidarApolloInstanceSegmentation::getRange() { return range_; }

int LidarApolloInstanceSegmentation::getWidth() { return width_; }

int LidarApolloInstanceSegmentation::getHeight() { return height_; }

std::vector<int> LidarApolloInstanceSegmentation::getCluster2dIdImage() { return cluster2d_->getIdImage(); }