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

#include "lidar_instance_segmentation_tvm/debugger.h"
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

Debugger::Debugger() : nh_(""), pnh_("~")
{
  instance_pointcloud_pub_ =
    pnh_.advertise<sensor_msgs::PointCloud2>("debug/instance_pointcloud", 1);
  confidence_image_pub_ = pnh_.advertise<sensor_msgs::Image>("debug/confidence_image", 1);
  category_image_pub_ = pnh_.advertise<sensor_msgs::Image>("debug/category_image", 1);
  class_image_pub_ = pnh_.advertise<sensor_msgs::Image>("debug/class_image", 1);
  instance_image_pub_ = pnh_.advertise<sensor_msgs::Image>("debug/instance_image", 1);
}

void Debugger::publishColoredPointCloud(
  const autoware_perception_msgs::DynamicObjectWithFeatureArray & input)
{
  pcl::PointCloud<pcl::PointXYZRGB> colored_pointcloud;
  for (size_t i = 0; i < input.feature_objects.size(); i++) {
    pcl::PointCloud<pcl::PointXYZI> object_pointcloud;
    pcl::fromROSMsg(input.feature_objects.at(i).feature.cluster, object_pointcloud);

    int red, green, blue;
    switch (input.feature_objects.at(i).object.semantic.type) {
      case autoware_perception_msgs::Semantic::CAR: {
        red = 255;
        green = 0;
        blue = 0;
        break;
      }
      // case autoware_perception_msgs::Semantic::TRUCK: {
      //   red = 255;
      //   green = 127;
      //   blue = 0;
      //   break;
      // }
      // case autoware_perception_msgs::Semantic::BUS: {
      //   red = 255;
      //   green = 0;
      //   blue = 127;
      //   break;
      // }
      case autoware_perception_msgs::Semantic::PEDESTRIAN: {
        red = 0;
        green = 255;
        blue = 255;
        // blue = 0;
        break;
      }
      // case autoware_perception_msgs::Semantic::BICYCLE: {
      //   red = 0;
      //   green = 0;
      //   blue = 255;
      //   break;
      // }
      case autoware_perception_msgs::Semantic::MOTORBIKE: {
        red = 0;
        // green = 127;
        green = 0;
        blue = 255;
        break;
      }
      case autoware_perception_msgs::Semantic::UNKNOWN: {
        red = 255;
        green = 255;
        blue = 255;
        break;
      }
    }

    for (size_t i = 0; i < object_pointcloud.size(); i++) {
      pcl::PointXYZRGB colored_point;
      colored_point.x = object_pointcloud[i].x;
      colored_point.y = object_pointcloud[i].y;
      colored_point.z = object_pointcloud[i].z;
      colored_point.r = red;
      colored_point.g = green;
      colored_point.b = blue;
      colored_pointcloud.push_back(colored_point);
    }
  }
  sensor_msgs::PointCloud2 output_msg;
  pcl::toROSMsg(colored_pointcloud, output_msg);
  output_msg.header = input.header;
  instance_pointcloud_pub_.publish(output_msg);
}

cv::Mat Debugger::getHeatMap(const float * feature, int height, int width)
{
  cv::Mat heatmap(height, width, CV_8UC1);
  for (int row = 0; row < height; ++row) {
    unsigned char * src = heatmap.ptr<unsigned char>(row);
    for (int col = 0; col < width; ++col) {
      int grid = row + col * width;
      if (*(feature + grid) > 1) {
        src[width - col - 1] = 255;
      } else if (*(feature + grid) > 0) {
        src[width - col - 1] = static_cast<int>(*(feature + grid) * 255);
      } else {
        src[width - col - 1] = 0;
      }
    }
  }
  return heatmap;
}

void Debugger::publishConfidenceImage(
  const std::shared_ptr<float> inferred_data, int height, int width,
  std_msgs::Header message_header)
{
  cv::Mat confidence_image(height, width, CV_8UC1);
  confidence_image = getHeatMap(inferred_data.get(), height, width);
  confidence_image_pub_.publish(
    cv_bridge::CvImage(message_header, sensor_msgs::image_encodings::MONO8, confidence_image)
      .toImageMsg());
}

void Debugger::publishCategoryImage(
  const std::shared_ptr<float> inferred_data, int height, int width,
  std_msgs::Header message_header)
{
  cv::Mat category_image(height, width, CV_8UC1);
  category_image = getHeatMap(inferred_data.get() + height * width * 3, height, width);
  category_image_pub_.publish(
    cv_bridge::CvImage(message_header, sensor_msgs::image_encodings::MONO8, category_image)
      .toImageMsg());
}

void Debugger::publishClassImage(
  const std::shared_ptr<float> inferred_data, int height, int width,
  std_msgs::Header message_header)
{
  cv::Mat class_image(height, width, CV_8UC3);

  for (int row = 0; row < height; ++row) {
    cv::Vec3b * src = class_image.ptr<cv::Vec3b>(row);
    for (int col = 0; col < width; ++col) {
      int grid = row + col * width;
      std::vector<float> class_vec{*(inferred_data.get() + grid + height * width * 4),
                                   *(inferred_data.get() + grid + height * width * 5),
                                   *(inferred_data.get() + grid + height * width * 6),
                                   *(inferred_data.get() + grid + height * width * 7),
                                   *(inferred_data.get() + grid + height * width * 8),
                                   *(inferred_data.get() + grid + height * width * 9)};
      std::vector<float>::iterator maxIt = std::max_element(class_vec.begin(), class_vec.end());
      size_t pred_class = std::distance(class_vec.begin(), maxIt);

      if (pred_class == 1) {
        src[width - col - 1] = cv::Vec3b(255, 0, 0);
      } else if (pred_class == 2) {
        src[width - col - 1] = cv::Vec3b(0, 0, 255);
      } else if (pred_class == 3) {
        src[width - col - 1] = cv::Vec3b(0, 255, 255);
      } else if (pred_class == 4) {
        src[width - col - 1] = cv::Vec3b(0, 0, 0);
      } else {
        src[width - col - 1] = cv::Vec3b(0, 0, 0);
      }
    }
  }

  class_image_pub_.publish(
    cv_bridge::CvImage(message_header, sensor_msgs::image_encodings::RGB8, class_image)
      .toImageMsg());
}

cv::Mat Debugger::colorizeIdImg(const std::vector<int> id_img, const int height, const int width)
{
  cv::Mat colorized_id_img(height, width, CV_8UC3);

  for (int row = 0; row < height; ++row) {
    cv::Vec3b * src = colorized_id_img.ptr<cv::Vec3b>(row);
    for (int col = 0; col < width; ++col) {
      int grid = row + col * width;
      int id = id_img[grid];
      if (id == -1) {
        src[width - col - 1] = cv::Vec3b(0, 0, 0);
      } else {
        int red = (id * 50) % 256;
        int green = (id * 70) % 256;
        int blue = (id * 90) % 256;
        src[width - col - 1] = cv::Vec3b(red, green, blue);
      }
    }
  }
  return colorized_id_img;
}

void Debugger::publishIdImage(
  const std::vector<int> id_img, int height, int width, std_msgs::Header message_header)
{
  cv::Mat instance_image = colorizeIdImg(id_img, height, width);
  instance_image_pub_.publish(
    cv_bridge::CvImage(message_header, sensor_msgs::image_encodings::RGB8, instance_image)
      .toImageMsg());
}