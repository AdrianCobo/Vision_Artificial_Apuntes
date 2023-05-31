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

#include <cmath>
#include <iostream>
#include <memory>
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"

#define PI 3.14159265
#define RAD_TO_DEGREES 180.0 / PI

const int RGB = (int)'1';
const int CMY = (int)'2';
const int HSI = (int)'3';
const int HSV = (int)'4';
const int HSV_CV = (int)'5';
const int HSI_CV = (int)'6';
const int NO_NEW_FORMAT = -1;

int CAMERA_MODE = RGB;

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
    sensor_msgs::msg::Image out_image;  // >> message to be sent
    img_bridge.toImageMsg(out_image);   // from cv_bridge to sensor_msgs::Image

    // Publish the data
    publisher_->publish(out_image);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

/**
  TO-DO
*/

cv::Mat bgr_2_cmy(cv::Mat src)
{
  // Read pixel values using split channels
  cv::Mat new_image;
  std::vector<cv::Mat> three_channels, channels;
  cv::split(src, three_channels);

  // Now I can access each channel separately
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // Transorm each pixel of the channels in order to get the CMY format
      // at the right part of the equality its important to remember that it is in BGR
      three_channels[0].at<uchar>(i, j) = 255 - (uint)three_channels[2].at<uchar>(i, j); 
      three_channels[1].at<uchar>(i, j) = 255 - (uint)three_channels[1].at<uchar>(i, j); 
      three_channels[2].at<uchar>(i, j) = 255 - (uint)three_channels[0].at<uchar>(i, j);
    }
  }

  // joing the channels in order to get a 3 channels img in CMY format
  channels.push_back(three_channels[0]);
  channels.push_back(three_channels[1]);
  channels.push_back(three_channels[2]);
  cv::merge(channels, new_image);

  return new_image;
}

cv::Mat bgr_2_hsi(cv::Mat src)
{
  double Hnum, Hden, B, G, R, H, S, I;
  cv::Mat RGBmin, new_image;
  std::vector<cv::Mat> three_channels, channels;

  // Read pixel values using split channels
  cv::split(src, three_channels);

  // get the minimum value of the 3 channels for each pixel
  cv::min(three_channels[0], three_channels[1], RGBmin);
  cv::min(RGBmin, three_channels[2], RGBmin);

  // Now I can access each channel separately
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      B = static_cast<double>(three_channels[0].at<uchar>(i, j));
      G = static_cast<double>(three_channels[1].at<uchar>(i, j));
      R = static_cast<double>(three_channels[2].at<uchar>(i, j));

      // normalize the values before processing
      B = B / 255.0;
      G = G / 255.0;
      R = R / 255.0;

      // Transorm each pixel of the channels in order to get the HSI format
      Hnum = 0.5 * ((R - G) + (R - B));
      Hden = sqrt(pow(R - B, 2.0) + (R - B) * (G - B));
      H = acos(Hnum / Hden) * RAD_TO_DEGREES;
      if (B > G) {
        H = 360.0 - H;
      }

      S = 1.0 - (3.0 / (R + G + B)) * (static_cast<double>(RGBmin.at<uchar>(i, j)) / 255.0);

      I = 1.0 / 3.0 * (R + G + B);

      // change the pixels channels value from 0-1 to 0-255
      three_channels[0].at<uchar>(i, j) = (uint)(H / 360.0 * 255.0);
      three_channels[1].at<uchar>(i, j) = (uint)(S * 255.0);
      three_channels[2].at<uchar>(i, j) = (uint)(I * 255.0);
    }
  }

  // joing the channels in order to get a 3 channels img in HSI format
  channels.push_back(three_channels[0]);
  channels.push_back(three_channels[1]);
  channels.push_back(three_channels[2]);
  cv::merge(channels, new_image);

  return new_image;
}

cv::Mat bgr_2_hsv(cv::Mat src)
{
  double Hnum, Hden, B, G, R, H, S, V;
  cv::Mat RGBmin, RGBmax, new_image;
  std::vector<cv::Mat> three_channels, channels;

  // Read pixel values using split channels
  cv::split(src, three_channels);

  // get the minimum value of the 3 channels for each pixel
  cv::min(three_channels[0], three_channels[1], RGBmin);
  cv::min(RGBmin, three_channels[2], RGBmin);

  // get the maximum value of the 3 channels for each pixel
  cv::max(three_channels[0], three_channels[1], RGBmax);
  cv::max(RGBmax, three_channels[2], RGBmax);

  // Now I can access each channel separately
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      B = static_cast<double>(three_channels[0].at<uchar>(i, j));
      G = static_cast<double>(three_channels[1].at<uchar>(i, j));
      R = static_cast<double>(three_channels[2].at<uchar>(i, j));

      // normalize the values before processing
      B = B / 255.0;
      G = G / 255.0;
      R = R / 255.0;

      // Transorm each pixel of the channels in order to get the HSV format
      Hnum = 0.5 * ((R - G) + (R - B));
      Hden = sqrt(pow(R - B, 2.0) + (R - B) * (G - B));
      H = acos(Hnum / Hden) * RAD_TO_DEGREES;
      if (B > G) {
        H = 360.0 - H;
      }

      S = 1.0 - (3.0 / (R + G + B)) * (static_cast<double>(RGBmin.at<uchar>(i, j)) / 255.0);

      V = static_cast<double>(RGBmax.at<uchar>(i, j));

      // change the pixels channels value from 0-1 to 0-255
      three_channels[0].at<uchar>(i, j) = (uint)(H / 360.0 * 255.0);
      three_channels[1].at<uchar>(i, j) = (uint)(S * 255.0);
      three_channels[2].at<uchar>(i, j) = (uint)V;
    }
  }

  // joing the channels in order to get a 3 channels img in HSV format
  channels.push_back(three_channels[0]);
  channels.push_back(three_channels[1]);
  channels.push_back(three_channels[2]);
  cv::merge(channels, new_image);

  return new_image;
}

// transorm a BGR image to an HSI image calculating manually the I channel
// and using the Opencv cvtColor function to calculate the H and S channel
cv::Mat bgr_2_hsi_manI(cv::Mat src)
{
  double B, G, R, I;
  cv::Mat RGBmax, new_image, temp;
  std::vector<cv::Mat> three_channels, three_channels_hsv, channels;

  // Read pixel values using split channels
  cv::cvtColor(src, temp, cv::COLOR_BGR2HSV);

  // Read pixel values using split channels
  cv::split(src, three_channels);
  cv::split(temp, three_channels_hsv);

  // Now I can access each channel separately
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      B = static_cast<double>(three_channels[0].at<uchar>(i, j));
      G = static_cast<double>(three_channels[1].at<uchar>(i, j));
      R = static_cast<double>(three_channels[2].at<uchar>(i, j));

      // normalize the values before processing
      B = B / 255.0;
      G = G / 255.0;
      R = R / 255.0;

      // Transorm each pixel of the channels in order to get the I channel format
      I = 1.0 / 3.0 * (R + G + B);

      // change the pixels channel value from 0-1 to 0-255
      three_channels_hsv[2].at<uchar>(i, j) = (uint)(I * 255.0);
    }
  }

  // joing the channels in order to get a 3 channels img in HSV format
  channels.push_back(three_channels_hsv[0]);
  channels.push_back(three_channels_hsv[1]);
  channels.push_back(three_channels_hsv[2]);
  cv::merge(channels, new_image);

  return new_image;
}

cv::Mat image_processing(const cv::Mat in_image)
{
  int key_pressed;
  cv::Mat out_image;
  cv::String text = "1: RGB, 2: CMY, 3: HSI, 4: HSV, 5: HSV Opencv, 6: HSI Opencv";

  // create output image
  out_image = in_image;

  switch (CAMERA_MODE) {
    // Option 1
    case RGB:
      break;
    // Option 2
    case CMY:
      out_image = bgr_2_cmy(out_image);
      break;
    // Option 3
    case HSI:
      out_image = bgr_2_hsi(out_image);
      break;
    // Option 4
    case HSV:
      out_image = bgr_2_hsv(out_image);
      break;
    // Option 5
    case HSV_CV:
      cvtColor(out_image, out_image, cv::COLOR_BGR2HSV);
      break;
    // Option 6
    case HSI_CV:
      out_image = bgr_2_hsi_manI(out_image);
      break;
  }

  // Write options text at the image
  cv::putText(
    out_image, text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  // show resultant image at window p1
  cv::imshow("p1", out_image);

  // wait new image format during 100ms
  key_pressed = cv::waitKey(100);

  // change the image format if it is necessary
  if (key_pressed != NO_NEW_FORMAT) {
    CAMERA_MODE = key_pressed;
  }

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}
