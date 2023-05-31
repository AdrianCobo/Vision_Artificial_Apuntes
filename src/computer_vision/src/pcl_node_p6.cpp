/*
# Copyright (c) 2022 José Miguel Guerrero Hernández
# Copyright (c) 2023 Adrián Cobo Merino
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include <pcl/ModelCoefficients.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <image_transport/image_transport.hpp>
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"

#define TF_HEADFRONT_BASE_X 0.047
#define TF_HEADFRONT_BASE_Y 1.067
#define TF_HEADFRONT_BASE_Z -0.216

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);

class PCLSubscriber : public rclcpp::Node
{
public:
  PCLSubscriber()
  : Node("opencv_subscriber")
  {
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos,
      std::bind(&PCLSubscriber::topic_callback_3d, this, std::placeholders::_1));

    publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcl_points", qos);
  }

private:
  void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
  {
    // Convert to PCL data type
    pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
    pcl::fromROSMsg(*msg, point_cloud);

    // PCL Processing
    pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud);

    // Convert to ROS data type
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(pcl_pointcloud, output);
    output.header = msg->header;

    // Publish the data
    publisher_3d_->publish(output);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_3d_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_3d_;
};

/**
  TO-DO
*/

pcl::PointCloud<pcl::PointXYZRGB> ball_filter(pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud, pink_pointcloud, pink_pointcloud_inliers;
  pcl::PointCloud<pcl::PointXYZHSV> hsv_pointcloud;

  pcl::PointCloudXYZRGBtoXYZHSV(in_pointcloud, hsv_pointcloud);

  for (size_t j = 0; j < hsv_pointcloud.size(); j++) {
    pcl::PointXYZHSV p = hsv_pointcloud.points[j];
    if (p.h >= 260 && p.h <= 340) {
      if (p.s > 0.2 && p.v > 0.2) {
        pink_pointcloud.push_back(in_pointcloud.points[j]);
      }
    }
  }

  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(pink_pointcloud));
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(pink_pointcloud_inliers);

  return pink_pointcloud_inliers;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_cube(
  pcl::PointCloud<pcl::PointXYZRGB> dst_pointcloud, pcl::PointXYZRGB center, double size,
  int r, int g, int b)
{
  pcl::PointCloud<pcl::PointXYZRGB> cube;

  for (double i = center.x - size / 2; i < center.x + size / 2; i += 0.01) {
    for (double j = center.y - size / 2; j < center.y + size / 2; j += 0.01) {
      for (double k = center.z - size / 2; k < center.z + size / 2; k += 0.01) {
        pcl::PointXYZRGB p = pcl::PointXYZRGB(i, j, k, r, g, b);
        cube.push_back(p);
      }
    }
  }

  cube = cube + dst_pointcloud;

  return cube;
}

pcl::SACSegmentation<pcl::PointXYZRGB> initialice_segmentator()
{
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;    // Create the segmentation object

  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_SPHERE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(0.01);
  return seg;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_sphere(pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZRGB>),
  cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::PointCloud<pcl::PointXYZRGB> result = in_pointcloud;
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ExtractIndices<pcl::PointXYZRGB> extract; // Create the filtering object
  float x, y, z;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr spheres_point_cloud(in_pointcloud.makeShared());
  int nr_points = (int)spheres_point_cloud->size();
  seg = initialice_segmentator();

  // While 30% of the original cloud is still there
  while (spheres_point_cloud->size() > 0.2 * nr_points) {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(spheres_point_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
      std::cerr << "Could not estimate a sphere model for the given dataset." << std::endl;
      break;
    }

    x = coefficients->values[0];
    y = coefficients->values[1];
    z = coefficients->values[2];

    pcl::PointXYZRGB p = pcl::PointXYZRGB(x, y, z, 0, 0, 0);
    result = draw_cube(result, p, 0.1, 0, 0, 255);

    // Extract the inliers
    extract.setInputCloud(spheres_point_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_p);

    // Create the filtering object
    extract.setNegative(true);
    extract.filter(*cloud_f);
    spheres_point_cloud.swap(cloud_f);
  }
  return result;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_floor_cubes(
  pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> result = in_pointcloud;
  int px, py, pz;

  for (int i = 3 + TF_HEADFRONT_BASE_Y; i <= 8 + TF_HEADFRONT_BASE_Y; i++) {
    px = TF_HEADFRONT_BASE_X + 1;
    py = TF_HEADFRONT_BASE_Y;
    pz = TF_HEADFRONT_BASE_Z + i;

    pcl::PointXYZRGB p1 = pcl::PointXYZRGB(-px, py, pz, 0, 0, 0);
    pcl::PointXYZRGB p2 = pcl::PointXYZRGB(px, py, pz, 0, 0, 0);
    result = draw_cube(result, p1, 0.1, 0, 255 * (i - 2) / 6, 0);
    result = draw_cube(result, p2, 0.1, 0, 255 * (i - 2) / 6, 0);
  }
  return result;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> pink_points, spheres, out_pointcloud;
     
  pink_points = ball_filter(in_pointcloud);
  spheres = pcl_sphere(pink_points);
  out_pointcloud = draw_floor_cubes(spheres);

  return out_pointcloud;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PCLSubscriber>());
  rclcpp::shutdown();
  return 0;
}
