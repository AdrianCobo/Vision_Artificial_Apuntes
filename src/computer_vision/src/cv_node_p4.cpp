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

#include <stdlib.h>
#include <time.h>

#include <cmath>
#include <image_transport/image_transport.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"

const int BGR = 0;          // option 1
const int GREEN_LINES = 1;  // option 2
const int BLUE_BALLS = 2;   // option 3
const int MOMENTS = 3;      // option 4

const int NO_NEW_FORMAT = -1;  // no key pressed
const int LOW_PASS_FILTER_RADIUS = 50;

const cv::Scalar RED = cv::Scalar(0, 0, 255);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
const cv::Scalar CYAN = cv::Scalar(255, 255, 0);
const cv::Scalar BLUE = cv::Scalar(255, 0, 0);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0);

const int MAX_OPTIONS = 3;
const int MAX_HOUGH = 200;
const int MAX_AREA = 1000;

const char * p4_window_name = "P4";
const char * trackbar_1_text = "0. Original; 1. Lines; 2. Balls; 3. Contours";
const char * trackbar_2_text = "Hough accumulator";
const char * trackbar_3_text = "Area";

// Definir rango de valores de color verde en formato HSV
const cv::Scalar LOWER_GREEN = cv::Scalar(40, 10, 10);
const cv::Scalar UPPER_GREEN = cv::Scalar(80, 255, 255);
const cv::Scalar LOWER_BLUE = cv::Scalar(80, 10, 10);
const cv::Scalar UPPER_BLUE = cv::Scalar(130, 255, 255);

int CAMERA_MODE = BGR;
bool initialized_img = false;
int hough, area;

cv::Mat image_processing(const cv::Mat in_image);

class ComputerVisionSubscriber : public rclcpp::Node
{
public:
  ComputerVisionSubscriber()
  : Node("opencv_subscriber")
  {
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos,
      std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("cv_image", qos);
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

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

/**
  TO-DO
*/

void read_key_pressed()
{
  int key_pressed;
  // wait new image format during 100ms
  key_pressed = cv::waitKey(100);

  // change the image format if it is necessary
  if (key_pressed != NO_NEW_FORMAT) {
    CAMERA_MODE = key_pressed;
  }
}

cv::Mat color_filter(cv::Mat src, cv::Scalar lower_range_color, cv::Scalar uper_range_color)
{
  cv::Mat hsv, mask_green, image_color_filtered;

  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  // Aplicar máscara para obtener solo los píxeles verdes
  cv::Mat color_mask;
  cv::inRange(hsv, lower_range_color, uper_range_color, color_mask);

  cv::bitwise_and(hsv, hsv, image_color_filtered, color_mask);

  return image_color_filtered;
}

cv::Mat color_mask(cv::Mat src, cv::Scalar lower_range_color, cv::Scalar uper_range_color)
{
  cv::Mat hsv, mask_green, color_mask;

  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  // Aplicar máscara para obtener solo los píxeles verdes
  cv::inRange(hsv, lower_range_color, uper_range_color, color_mask);

  return color_mask;
}

cv::Mat print_detected_lines(const cv::Mat src, const std::vector<cv::Vec2f> lines)
{
  cv::Mat img_lines_printed = src.clone();

  // Draw the lines
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));

    cv::Point line_center((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
    cv::line(img_lines_printed, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    cv::circle(img_lines_printed, line_center, 1, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
  }
  return img_lines_printed;
}

cv::Mat print_detected_circles(const cv::Mat src, const std::vector<cv::Vec3f> circles)
{
  cv::Mat img_circles_printed = src.clone();

  for (size_t i = 0; i < circles.size(); i++) {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    // circle center
    cv::circle(img_circles_printed, center, 1, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    // circle outline
    int radius = c[2];
    cv::circle(img_circles_printed, center, radius, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
  }

  return img_circles_printed;
}

void initialize_user_interface()
{
  cv::namedWindow(p4_window_name, 0);

  // create Trackbar and add to a window
  cv::createTrackbar(trackbar_1_text, p4_window_name, nullptr, MAX_OPTIONS, 0);
  cv::createTrackbar(trackbar_2_text, p4_window_name, nullptr, MAX_HOUGH, 0);
  cv::createTrackbar(trackbar_3_text, p4_window_name, nullptr, MAX_AREA, 0);

  // set Trackbar’s value
  cv::setTrackbarPos(trackbar_1_text, p4_window_name, 0);
  cv::setTrackbarPos(trackbar_2_text, p4_window_name, 0);
  cv::setTrackbarPos(trackbar_3_text, p4_window_name, 0);

  srand(time(NULL));
  initialized_img = true;
}

cv::Mat detect_green_lines(const cv::Mat src)
{
  cv::Mat green_filtered, canny, out_image;
  std::vector<cv::Vec2f> green_lines;

  green_filtered = color_filter(src, LOWER_GREEN, UPPER_GREEN);
  // Edge detection
  cv::Canny(green_filtered, canny, 50, 200, 3);

  // Standard Hough Line Transform
  cv::HoughLines(canny, green_lines, 1, CV_PI / 180, hough, 0, 0);    // runs the actual detection

  return out_image = print_detected_lines(src, green_lines);
}

cv::Mat detect_blue_balls(const cv::Mat src)
{
  cv::Mat blue_filtered, auxiliar_bgr, auxiliar_gray, out_image;
  std::vector<cv::Vec3f> circles;

  blue_filtered = color_filter(src, LOWER_BLUE, UPPER_BLUE);

  cv::medianBlur(blue_filtered, blue_filtered, 9);
  cv::cvtColor(blue_filtered, auxiliar_bgr, cv::COLOR_HSV2BGR);
  cv::cvtColor(auxiliar_bgr, auxiliar_gray, cv::COLOR_BGR2GRAY);

  // change 5th argumentto detect circles with different distances to each other
  // change the last two parameters (min_radius & max_radius) to detect larger circles
  cv::HoughCircles(
    auxiliar_gray, circles, cv::HOUGH_GRADIENT, 1, auxiliar_gray.rows / 16, 160.0, 12.0, 15,
    77);

  return out_image = print_detected_circles(src, circles);
}

cv::Mat draw_contours(
  const cv::Mat src, std::vector<std::vector<cv::Point>> contours,
  std::vector<cv::Vec4i> hierarchy)
{
  cv::Mat out_image = src;
  int idx = 0;

  while (idx >= 0 && contours.size() > 0) {
    cv::Scalar color(255, 0, 0);
    // Last value navigates into the hierarchy
    cv::drawContours(out_image, contours, idx, color, cv::FILLED, 8, hierarchy, 1);
    idx = hierarchy[idx][0];
  }

  // draw center of the moment
  for (size_t i = 0; i < contours.size(); i++) {
    double contour_area = cv::contourArea(contours[i]);
    int r, g, b;
    r = std::rand() % 256;
    g = std::rand() % 256;
    b = std::rand() % 256;
    if (contour_area > (double)area) {
      cv::Moments moments = cv::moments(contours[i]);
      cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
      cv::Point center_text(moments.m10 / moments.m00 + 10, moments.m01 / moments.m00);
      cv::circle(out_image, center, 5, cv::Scalar(0, 0, 255), -1);
      cv::putText(
        out_image, std::to_string(contour_area), center_text, cv::FONT_HERSHEY_SIMPLEX, 1,
        cv::Scalar(b, g, r), 2);
    }
  }
  return out_image;
}

cv::Mat image_processing(const cv::Mat in_image)
{
  cv::Mat noise_filtered_img, green_filtered, canny, blue_filtered, both_colors_img_filtered,
    mask_aplied;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  if (!initialized_img) {
    initialize_user_interface();
  }

  // refresh the global paramiters
  CAMERA_MODE = cv::getTrackbarPos(trackbar_1_text, p4_window_name);
  hough = cv::getTrackbarPos(trackbar_2_text, p4_window_name);
  area = cv::getTrackbarPos(trackbar_3_text, p4_window_name);

  cv::Mat out_image = in_image;
  cv::medianBlur(in_image, noise_filtered_img, 3);

  switch (CAMERA_MODE) {
    case GREEN_LINES:
      out_image = detect_green_lines(noise_filtered_img);
      break;

    case BLUE_BALLS:
      out_image = detect_blue_balls(noise_filtered_img);
      break;

    case MOMENTS:
      green_filtered = color_mask(noise_filtered_img, LOWER_GREEN, UPPER_GREEN);
      blue_filtered = color_mask(noise_filtered_img, LOWER_BLUE, UPPER_BLUE);
      both_colors_img_filtered = green_filtered + blue_filtered;
      cv::bitwise_and(
        noise_filtered_img, noise_filtered_img, mask_aplied, both_colors_img_filtered);

      cv::Canny(mask_aplied, canny, 50, 150, 3);
      cv::findContours(canny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

      out_image = draw_contours(out_image, contours, hierarchy);
      break;

    case BGR:
    default:
      break;
  }

  // show resultant image at window P3
  cv::imshow(p4_window_name, out_image);
  read_key_pressed();
  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}
