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

const int BGR = (int)'1';                 // key 1
const int GRAY = (int)'2';                // key 2
const int ENHANCED = (int)'3';            // key 3
const int SHRINK_MIN_GROW = (int)'x';    // key x
const int SHRINK_MIN_REDUCE = (int)'z';  // key z
const int SHRINK_MAX_REDUCE = (int)'c';    // key c
const int SHRINK_MAX_GROW = (int)'v';   // key v
const int NO_NEW_FORMAT = -1;       // no key pressed
const int LOW_PASS_FILTER_RADIUS = 50;

const cv::Scalar RED = cv::Scalar(0, 0, 255);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
const cv::Scalar CYAN = cv::Scalar(255, 255, 0);
const cv::Scalar BLUE = cv::Scalar(255, 0, 0);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0);

double shrink_min = 0.0;
double shrink_max = 30.0;

int RESULT_COMP_STRING_IDX = 0;
int CAMERA_MODE = BGR;
int PREV_CAMERA_MODE; // ALLOW US TO NOT CHANGE THE IMAGE WHILE CHANGING SHRINK VALUES
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

// Compute the Discrete fourier transform
cv::Mat computeDFT(const cv::Mat & image)
{
  // Expand the image to an optimal size.
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(image.rows);
  int n = cv::getOptimalDFTSize(image.cols);    // on the border add zero values
  cv::copyMakeBorder(
    image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT,
    cv::Scalar::all(0));

  // Make place for both the complex and the real values
  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);    // Add to the expanded another plane with zeros

  // Make the Discrete Fourier Transform
  cv::dft(
    complexI, complexI,
    cv::DFT_COMPLEX_OUTPUT);      // this way the result may fit in the source matrix
  return complexI;
}

// 6. Crop and rearrange
cv::Mat fftShift(const cv::Mat & magI)
{
  cv::Mat magI_copy = magI.clone();
  // crop the spectrum, if it has an odd number of rows or columns
  magI_copy = magI_copy(cv::Rect(0, 0, magI_copy.cols & -2, magI_copy.rows & -2));

  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI_copy.cols / 2;
  int cy = magI_copy.rows / 2;

  cv::Mat q0(magI_copy, cv::Rect(0, 0, cx, cy));      // Top-Left - Create a ROI per quadrant
  cv::Mat q1(magI_copy, cv::Rect(cx, 0, cx, cy));     // Top-Right
  cv::Mat q2(magI_copy, cv::Rect(0, cy, cx, cy));     // Bottom-Left
  cv::Mat q3(magI_copy, cv::Rect(cx, cy, cx, cy));    // Bottom-Right

  cv::Mat tmp;    // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return magI_copy;
}

cv::Mat apply_low_pass_filter(const cv::Mat & img)
{
  cv::Mat rearrange;

  // Compute the Discrete fourier transform
  cv::Mat complexImg = computeDFT(img);

  // Rearrange quadrants - Spectrum with low values at center - Theory mode
  cv::Mat shift_complex = fftShift(complexImg);

  // mask for low_pass_filter
  cv::Mat mask = cv::Mat::zeros(shift_complex.rows, shift_complex.cols, shift_complex.type());

  // low pass mask patron
  cv::circle(
    mask, cv::Point(mask.cols / 2, mask.rows / 2), LOW_PASS_FILTER_RADIUS,
    cv::Scalar(1.0f, 1.0f, 1.0f), -1);

  // apply filter
  cv::mulSpectrums(shift_complex, mask, shift_complex, 0);

  // Rearrange quadrants - Spectrum with low
  // values at corners - OpenCV mode
  rearrange = fftShift(shift_complex);

  cv::Mat inverse_transform;
  cv::idft(rearrange, inverse_transform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(inverse_transform, inverse_transform, 0, 1, cv::NORM_MINMAX);

  return inverse_transform;
}

void read_key_pressed()
{
  int key_pressed;
  // wait new image format during 100ms
  key_pressed = cv::waitKey(100);

  // change the image format if it is necessary
  if (key_pressed != NO_NEW_FORMAT) {
    PREV_CAMERA_MODE = CAMERA_MODE;
    CAMERA_MODE = key_pressed;
  }
}

cv::Mat contract_histogram_img(const cv::Mat & img)
{
  cv::Mat out_img = img.clone();
  // Read pixel values
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      out_img.at<uchar>(i, j) = (double)(shrink_max - shrink_min) / (255.0 - 0.0) *
        ((double)img.at<uchar>(i, j) - 0.0) +
        shrink_min;
    }
  }
  return out_img;
}

cv::Mat expand_histogram_img(const cv::Mat & img)
{
  cv::Mat out_img = img.clone();
  double min, max;
  cv::minMaxLoc(img, &min, &max);
  // Read pixel values
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      out_img.at<uchar>(i, j) =
        (((double)img.at<uchar>(i, j) - min) / (max - min)) * (255.0 - 0.0) + 0.0;
    }
  }
  return out_img;
}

cv::Mat write_user_text_at_img(cv::Mat out_image)
{
  cv::String text_1 =
    "1: Original, 2: Gray, 3: Histograma | shrink: [z,x]: -+ min | [c,v]: -+ max";
  cv::String text_2 = "shrink [min: 0, max: 30]";

  cv::rectangle(
    out_image, cv::Point2d(0, 0), cv::Point2d(out_image.cols, 40), cv::Scalar(255, 255, 255),
    -1);

  // Write options text at the image
  cv::putText(
    out_image, text_1, cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255),
    2);
  cv::putText(
    out_image, text_2, cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255),
    2);

  return out_image;
}

cv::Mat print_hist(const cv::Mat in_image, cv::Mat histImage, cv::Scalar color)
{
  cv::Mat hist_gray;
  // Initialize the arguments to calculate the histograms (bins, ranges and channels H and S )
  int histSize = 256;
  // hue varies from 0 to 179, saturation from 0 to 255
  float range[] = {0, 256};    // the upper boundary is exclusive
  const float * histRange = {range};
  bool uniform = true, accumulate = false;

  cv::calcHist(
    &in_image, 1, 0, cv::Mat(), hist_gray, 1, &histSize, &histRange, uniform, accumulate);

  int bin_w = cvRound((double)histImage.cols / histSize);
  // normalize the histograms between 0 and histImage.rows
  cv::normalize(hist_gray, hist_gray, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  // Draw the intensity line for histograms
  for (int i = 1; i < histSize; i++) {
    cv::line(
      histImage,
      cv::Point(bin_w * (i - 1), histImage.rows - cvRound(hist_gray.at<float>(i - 1))),
      cv::Point(bin_w * (i), histImage.rows - cvRound(hist_gray.at<float>(i))), color, 2, 8,
      0);
  }
  return histImage;
}

double compare_hist(cv::Mat src1, const cv::Mat src2)
{
  cv::Mat hist_1, hist_2;
  // Initialize the arguments to calculate the histograms (bins, ranges and channels H and S )
  int histSize = 256;
  // hue varies from 0 to 179, saturation from 0 to 255
  float range[] = {0, 256};    // the upper boundary is exclusive
  const float * histRange = {range};
  bool uniform = true, accumulate = false;

  cv::calcHist(&src1, 1, 0, cv::Mat(), hist_1, 1, &histSize, &histRange, uniform, accumulate);
  cv::calcHist(&src2, 1, 0, cv::Mat(), hist_2, 1, &histSize, &histRange, uniform, accumulate);

  cv::normalize(hist_1, hist_1, 0, hist_1.rows, cv::NORM_MINMAX, -1, cv::Mat());
  cv::normalize(hist_2, hist_2, 0, hist_2.rows, cv::NORM_MINMAX, -1, cv::Mat());

  double comparation = cv::compareHist(hist_1, hist_2, CV_COMP_CORREL);
  return comparation;
}

void print_result_hist_comp(cv::Mat hist, double result, cv::Point loc, cv::Scalar color)
{
  cv::String possible_results[4];
  possible_results[0] = "Shrink [" + std::to_string((int)shrink_min) + "," +
    std::to_string((int)shrink_max) + ": " + std::to_string(result);
  possible_results[1] = "Substract: " + std::to_string(result);

  possible_results[2] = "Stretch: " + std::to_string(result);
  possible_results[3] = "Eqhist: " + std::to_string(result);

  cv::putText(
    hist, possible_results[RESULT_COMP_STRING_IDX % 4], loc, cv::FONT_HERSHEY_SIMPLEX, 0.35,
    color, 2);
  RESULT_COMP_STRING_IDX += 1;
}

void shrink_min_grow()
{
  if (shrink_min < shrink_max - 1.0) {
    shrink_min += 1.0;
  }

  std::cout << "shrink_min: " << shrink_min << std::endl;
}

void shrink_min_reduce()
{
  if (shrink_min <= shrink_max - 1.0 && shrink_min > 0) {
    shrink_min -= 1.0;
  }

  std::cout << "shrink_min: " << shrink_min << std::endl;
}

void shrink_max_grow()
{
  if (shrink_min <= shrink_max - 1.0 && shrink_max < 255) {
    shrink_max += 1.0;
  }

  std::cout << "shrink_max: " << shrink_max << std::endl;
}

void shrink_max_reduce()
{
  if (shrink_min < shrink_max - 1.0) {
    shrink_max -= 1.0;
  }

  std::cout << "shrink_max: " << shrink_max << std::endl;
}

cv::Mat image_processing(const cv::Mat in_image)
{
  cv::Mat noise_filtered_img, gray_img, low_pass_filter_img, contracted_hist_img, substracted_img,
    expanded_hist_img;

  cv::Mat hist;
  int hist_w = 512, hist_h = 400;
  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
  double result;

  cv::Mat out_image = in_image;
  cv::medianBlur(in_image, noise_filtered_img, 3);

  // with this the image wont change while we configure the shrink values
  switch (CAMERA_MODE) {
    case SHRINK_MIN_GROW:
      shrink_min_grow();
      CAMERA_MODE = PREV_CAMERA_MODE;
      break;

    case SHRINK_MIN_REDUCE:
      shrink_min_reduce();
      CAMERA_MODE = PREV_CAMERA_MODE;
      break;

    case SHRINK_MAX_GROW:
      shrink_max_grow();
      CAMERA_MODE = PREV_CAMERA_MODE;
      break;

    case SHRINK_MAX_REDUCE:
      shrink_max_reduce();
      CAMERA_MODE = PREV_CAMERA_MODE;
      break;
  }

  switch (CAMERA_MODE) {
    case GRAY:
      cv::cvtColor(noise_filtered_img, out_image, cv::COLOR_BGR2GRAY);
      break;

    case ENHANCED:
      // step 1
      cv::cvtColor(noise_filtered_img, gray_img, cv::COLOR_BGR2GRAY);
      hist = print_hist(gray_img, histImage, BLUE);

      // step 2
      low_pass_filter_img = apply_low_pass_filter(gray_img);

      // step 3
      low_pass_filter_img = 255 * low_pass_filter_img;
      low_pass_filter_img.convertTo(low_pass_filter_img, CV_8UC1);
      contracted_hist_img = contract_histogram_img(low_pass_filter_img);
      hist = print_hist(contracted_hist_img, histImage, RED);
      result = compare_hist(gray_img, contracted_hist_img);
      print_result_hist_comp(hist, result, cv::Point(10, 15), RED);

      // step 4
      substracted_img = gray_img - contracted_hist_img;
      hist = print_hist(substracted_img, histImage, CYAN);
      result = compare_hist(gray_img, substracted_img);
      print_result_hist_comp(hist, result, cv::Point(10, 30), CYAN);

      // step 5
      expanded_hist_img = expand_histogram_img(substracted_img);
      hist = print_hist(expanded_hist_img, histImage, YELLOW);
      result = compare_hist(gray_img, expanded_hist_img);
      print_result_hist_comp(hist, result, cv::Point(10, 45), YELLOW);

      // final step
      cv::equalizeHist(expanded_hist_img, out_image);
      hist = print_hist(out_image, histImage, GREEN);
      result = compare_hist(gray_img, out_image);
      print_result_hist_comp(hist, result, cv::Point(10, 60), GREEN);

      cv::imshow("Histograms", hist);
      break;

    case BGR:
    default:
      break;
  }

  cv::Mat final_img = write_user_text_at_img(out_image);

  // show resultant image at window P3
  cv::imshow("P3", final_img);
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
