/*
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

#include <image_geometry/pinhole_camera_model.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <functional>
#include <image_transport/image_transport.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

using namespace std::chrono_literals;

const int CLEAN = 0;     // option 1
const int OPTION_1 = 1;  // option 2
const int OPTION_2 = 2;  // option 3

const cv::Scalar RED = cv::Scalar(0, 0, 255);
const cv::Scalar WHITE = cv::Scalar(255, 255, 255);

const int MAX_OPTIONS = 2;
const int MAX_ITERATIONS = 200;
const int MAX_DISTANCE = 8;

const char * p5_window_name = "P5";
const char * trackbar_1_text = "Option";
const char * trackbar_2_text = "Iterations";
const char * trackbar_3_text = "Distance";

// Definir rango de valores de color verde en formato HSV
const cv::Scalar LOWER_RED_1 = cv::Scalar(160, 60, 60);
const cv::Scalar UPPER_RED_1 = cv::Scalar(180, 255, 255);
const cv::Scalar LOWER_RED_2 = cv::Scalar(0, 60, 60);
const cv::Scalar UPPER_RED_2 = cv::Scalar(9, 255, 255);
const cv::Scalar LOWER_WHITE = cv::Scalar(0, 0, 100);
const cv::Scalar UPPER_WHITE = cv::Scalar(180, 0, 120);

int CAMERA_MODE = CLEAN;
int TRACKBAR_ITERATIONS, TRACKBAR_DISTANCE;

bool initialized_img = false;

cv::Mat depth_image;  // normal depth img

image_geometry::PinholeCameraModel intrinsic_camera_matrix;
geometry_msgs::msg::TransformStamped camera2basefootprint;

std::vector<cv::Point> Points;  // list of the cv::Points pixels clicked

cv::Mat image_processing(const cv::Mat in_image);

class ComputerVisionSubscriber : public rclcpp::Node
{
public:
  ComputerVisionSubscriber()
  : Node("opencv_subscriber"), tf_buffer_(), tf_listener_(tf_buffer_)
  {
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos,
      std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));

    subscription_dist_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/depth_registered/image_raw", qos,
      std::bind(
        &ComputerVisionSubscriber::distance_image_callback, this, std::placeholders::_1));

    subscription_camera_intrinsic_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/head_front_camera/depth_registered/camera_info", qos,
      std::bind(
        &ComputerVisionSubscriber::intrinsic_params_callback, this, std::placeholders::_1));

    // Call on_timer function every 500ms in order to get our tf
    timer_ = create_wall_timer(500ms, std::bind(&ComputerVisionSubscriber::on_timer, this));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("cv_image", qos);
  }

  void on_timer()
  {
    try {
      camera2basefootprint = tf_buffer_.lookupTransform(
        "head_front_camera_depth_optical_frame", "base_footprint", tf2::TimePointZero);
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Obstacle transform not found: %s", ex.what());
      return;
    }
  }

private:
  void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
  {
    // Convert ROS Image to CV Image
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_raw = cv_ptr->image;

    // Image processing
    cv::Mat cv_image = image_processing(image_raw);
    // Convert OpenCV Image to ROS Image
    cv_bridge::CvImage img_bridge =
      cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
    sensor_msgs::msg::Image out_image;      // >> message to be sent
    img_bridge.toImageMsg(out_image);       // from cv_bridge to sensor_msgs::Image

    // Publish the data
    publisher_->publish(out_image);
  }

  void distance_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
  {
    // Convertir los datos de profundidad a un objeto Mat de OpenCV
    cv_bridge::CvImagePtr depth_image_ptr =
      cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);

    depth_image = depth_image_ptr->image;
  }

  void intrinsic_params_callback(const sensor_msgs::msg::CameraInfo msg) const
  {
    intrinsic_camera_matrix = image_geometry::PinholeCameraModel();
    intrinsic_camera_matrix.fromCameraInfo(msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_dist_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription_camera_intrinsic_;
  rclcpp::TimerBase::SharedPtr timer_;
  tf2::BufferCore tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string target_frame_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

/**
  TO-DO
*/

cv::Mat color_mask(cv::Mat src, cv::Scalar lower_range_color, cv::Scalar uper_range_color)
{
  cv::Mat hsv, mask_green, color_mask;

  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  // Aplicar máscara para obtener solo los píxeles verdes
  cv::inRange(hsv, lower_range_color, uper_range_color, color_mask);

  return color_mask;
}

// Apply the thinning procedure to a given image
cv::Mat thinning(cv::InputArray input)
{
  cv::Mat src = input.getMat().clone();
  cv::Mat skeleton, temp, open, erode;
  cv::Mat element = cv::getStructuringElement(0, cv::Size(3, 3), cv::Point(1, 1));
  cv::Mat element_2 = cv::getStructuringElement(0, cv::Size(9, 9), cv::Point(3, 3));

  cv::dilate(src, src, element_2);    // fill black spaces in the filter

  skeleton = cv::Mat::zeros(src.size(), src.type());

  for (int i = 0; i < TRACKBAR_ITERATIONS; i++) {
    cv::morphologyEx(src, open, 2, element);
    temp = src - open;
    cv::erode(src, src, element);
    cv::bitwise_or(temp, skeleton, skeleton);
  }

  return skeleton;
}

cv::Point proyect_3d_to_2d(double px, double py, double pz)
{
  double x, y, z;
  uint x_2d, y_2d;

  x = camera2basefootprint.transform.translation.x;
  y = camera2basefootprint.transform.translation.y;
  z = camera2basefootprint.transform.translation.z;

  // create ou rototraslation matrix with no rotation in this case because we ae goint to use the
  // refence axis from camera frame instead of the base_footprint frame coordinates
  cv::Mat rot_tras_mat =
    (cv::Mat_<double>(3, 4) << 1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z);
  cv::Mat p_3d = (cv::Mat_<double>(4, 1) << px, py, pz, 1.0);
  cv::Mat p_2d = intrinsic_camera_matrix.intrinsicMatrix() * (rot_tras_mat * p_3d);

  // normalize the 2d pixel coordinates
  x_2d = (uint)(p_2d.at<double>(0, 0) / p_2d.at<double>(2, 0));
  y_2d = (uint)(p_2d.at<double>(1, 0) / p_2d.at<double>(2, 0));

  return cv::Point(x_2d, y_2d);
}

cv::Mat print_3d_to_2d_proyection_lines(cv::Mat input)
{
  for (int i = 0; i <= TRACKBAR_DISTANCE; i++) {
    cv::Point p1 = proyect_3d_to_2d(1.4, 0.0, double(i));
    cv::Point p2 = proyect_3d_to_2d(-1.4, 0.0, double(i));

    cv::line(input, p1, p2, cv::Scalar(80, 30 * i, 80), 2);
    p1.x = p1.x + 10;
    cv::putText(
      input, std::to_string(i), p1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 30 * i, 80),
      2);
  }

  return input;
}

// Apply the thinning procedure to a given image
cv::Mat print_3d_to_2d_proyection_points(cv::Mat input)
{
  double fx, fy, cx, cy, x_3d, y_3d, z_3d, d;
  int px, py;

  cv::Mat intrinsic_marix = (cv::Mat)intrinsic_camera_matrix.intrinsicMatrix();

  fx = intrinsic_marix.at<double>(0, 0);
  fy = intrinsic_marix.at<double>(1, 1);
  cx = intrinsic_marix.at<double>(0, 2);
  cy = intrinsic_marix.at<double>(1, 2);

  for (size_t i = 0; i < Points.size(); i++) {
    px = Points[i].x;
    py = Points[i].y;
    d = depth_image.at<float>(py, px);

    x_3d = ((double)px - cx) * d / fx;
    y_3d = ((double)py - cy) * d / fy;
    z_3d = d;

    std::string coordinates = "[";
    coordinates.append(
      std::to_string(x_3d) + " " + std::to_string(y_3d) + " " + std::to_string(z_3d) + "]");

    cv::circle(input, Points[i], 2, WHITE);
    cv::putText(
      input, coordinates, cv::Point(px + 10, py), cv::FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2);
  }

  return input;
}

// create mouse callback
void on_mouse(int event, int x, int y, int, void *)
{
  if (event == cv::EVENT_LBUTTONDOWN) {
    Points.push_back(cv::Point(x, y));
  }
}

void initialize_user_interface()
{
  cv::namedWindow(p5_window_name, 0);

  // create Trackbar and add to a window
  cv::createTrackbar(trackbar_1_text, p5_window_name, nullptr, MAX_OPTIONS, 0);
  cv::createTrackbar(trackbar_2_text, p5_window_name, nullptr, MAX_ITERATIONS, 0);
  cv::createTrackbar(trackbar_3_text, p5_window_name, nullptr, MAX_DISTANCE, 0);

  // set Trackbar’s value
  cv::setTrackbarPos(trackbar_1_text, p5_window_name, 0);
  cv::setTrackbarPos(trackbar_2_text, p5_window_name, 0);
  cv::setTrackbarPos(trackbar_3_text, p5_window_name, 0);

  cv::setMouseCallback(p5_window_name, on_mouse, 0);
  initialized_img = true;
}

cv::Mat detect_circuit_lines(const cv::Mat src)
{
  cv::Mat red_mask_1, red_mask_2, white_mask, circuit_lines, skeleton, out_image;

  red_mask_1 = color_mask(src, LOWER_RED_1, UPPER_RED_1);
  red_mask_2 = color_mask(src, LOWER_RED_2, UPPER_RED_2);
  white_mask = color_mask(src, LOWER_WHITE, UPPER_WHITE);

  circuit_lines = red_mask_1 + red_mask_2 + white_mask;

  if (TRACKBAR_DISTANCE >= 3) {
    cv::Point skeleton_limit = proyect_3d_to_2d(0.0, 0.0, (double)TRACKBAR_DISTANCE);
    cv::rectangle(
      circuit_lines, cv::Point(0, circuit_lines.rows),
      cv::Point(circuit_lines.cols, skeleton_limit.y), cv::Scalar(0), cv::FILLED);
  }

  skeleton = thinning(circuit_lines);
  out_image = src;

  for (int i = 0; i < skeleton.cols; i++) {
    for (int j = 0; j < skeleton.rows; j++) {
      uchar intensity = skeleton.at<uchar>(j, i);
      if (intensity != 0) {
        out_image.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 255, 0);
      }
    }
  }

  return out_image;
}

cv::Mat get_normaliced_depth_img()
{
  double min, max;
  cv::Mat depth_image_normalized;

  cv::patchNaNs(depth_image, 0.0);    // delete nans

  // delete infs
  for (int i = 0; i < depth_image.cols; i++) {
    for (int j = 0; j < depth_image.rows; j++) {
      if (isinf(depth_image.at<float>(j, i))) {
        depth_image.at<float>(j, i) = 0.0f;
      }
    }
  }

  // normalice the pixel values to [0, 255] range in order to get a correct display
  cv::minMaxLoc(depth_image, &min, &max);
  depth_image_normalized = (depth_image - min) / (max - min) * 255;
  depth_image_normalized.convertTo(depth_image_normalized, CV_8UC1);
  return depth_image_normalized;
}

cv::Mat image_processing(const cv::Mat in_image)
{
  cv::Mat noise_filtered_img, detected_circuit_lines, proyection_lines_printed,
    depth_image_normalized, out_image;

  if (!initialized_img) {
    initialize_user_interface();
  }

  // refresh the global paramiters
  CAMERA_MODE = cv::getTrackbarPos(trackbar_1_text, p5_window_name);
  TRACKBAR_ITERATIONS = cv::getTrackbarPos(trackbar_2_text, p5_window_name);
  TRACKBAR_DISTANCE = cv::getTrackbarPos(trackbar_3_text, p5_window_name);

  cv::medianBlur(in_image, noise_filtered_img, 3);

  switch (CAMERA_MODE) {
    case OPTION_1:
      detected_circuit_lines = detect_circuit_lines(noise_filtered_img);
      proyection_lines_printed = print_3d_to_2d_proyection_lines(detected_circuit_lines);
      out_image = print_3d_to_2d_proyection_points(proyection_lines_printed);
      break;

    case OPTION_2:
      depth_image_normalized = get_normaliced_depth_img();
      out_image = print_3d_to_2d_proyection_points(depth_image_normalized);
      break;

    case CLEAN:
    default:
      out_image = in_image;
      Points.clear();
      break;
  }

  // show resultant image at window P3
  cv::imshow(p5_window_name, out_image);
  cv::waitKey(100);
  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}
