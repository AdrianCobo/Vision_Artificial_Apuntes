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

const int GRAY = (int)'1';                               // key 1
const int FOURIER_SPECTRUM = (int)'2';                   // key 2
const int JUST_VERTICAL_AND_HORIZONTAL_FREQ = (int)'3';  // key 3
const int NOT_VERTICAL_AND_HORIZONTAL_FREQ = (int)'4';   // key 4
const int UMBRAL = (int)'5';                             // key 5
const int FILTER_GROW = (int)'x';                       // key z
const int FILTER_REDUCE = (int)'z';                     // key x
const int DISPLAY_FILTERS = (int)'d';                   // key d
const int NO_NEW_FORMAT = -1;                      // no key pressed

int CAMERA_MODE = GRAY;
float filter_size = 50.0f;
// bool condition wich will help us controlling the images showed at option 5
bool threshold_img_showed = false;
bool initialized_img = false;
const char * p2_window_name = "P2";
cv::Mat umbralized_1, umbralized_2, spectrum_filter_1, spectrum_filter_2;

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

void initialize_user_interface()
{
  cv::namedWindow(p2_window_name, 0);
  initialized_img = true;
}

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

// Calculate dft spectrum
cv::Mat spectrum(const cv::Mat & complexI)
{
  cv::Mat complexImg = complexI.clone();
  // Shift quadrants
  cv::Mat shift_complex = fftShift(complexImg);

  // Transform the real and complex values to magnitude
  // compute the magnitude and switch to logarithmic scale
  // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  cv::Mat planes_spectrum[2];

  cv::split(shift_complex, planes_spectrum);    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

  // planes[0] = magnitude
  cv::magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);

  cv::Mat spectrum = planes_spectrum[0];

  // Switch to a logarithmic scale
  spectrum += cv::Scalar::all(1);
  cv::log(spectrum, spectrum);

  // Normalize:
  // viewable image form (float between values 0 and 1).
  // Transform the matrix with float values into a
  cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX);

  return spectrum;
}

// Draw 4 black rectangles will keep the central cross of the image keeping the vertical
// and horizontal frequencies
cv::Mat apply_keep_filter(const cv::Mat & img)
{
  cv::Mat shift_complex = img;

  cv::rectangle(
    shift_complex, cv::Point2d(0, 0),
    cv::Point2d(
      shift_complex.cols / 2 - filter_size / 2, shift_complex.rows / 2 - filter_size / 2),
    cv::Scalar(0.0f, 0.0f, 0.0f), -1);
  cv::rectangle(
    shift_complex, cv::Point2d(shift_complex.cols / 2 + filter_size / 2, 0),
    cv::Point2d(shift_complex.cols, shift_complex.rows / 2 - filter_size / 2),
    cv::Scalar(0.0f, 0.0f, 0.0f), -1);
  cv::rectangle(
    shift_complex, cv::Point2d(0, shift_complex.rows),
    cv::Point2d(
      shift_complex.cols / 2 - filter_size / 2, shift_complex.rows / 2 + filter_size / 2),
    cv::Scalar(0.0f, 0.0f, 0.0f), -1);
  cv::rectangle(
    shift_complex, cv::Point2d(shift_complex.cols / 2 + filter_size / 2, shift_complex.rows),
    cv::Point2d(shift_complex.cols, shift_complex.rows / 2 + filter_size / 2),
    cv::Scalar(0.0f, 0.0f, 0.0f), -1);

  return shift_complex;
}

// Draw 2 black rectangles creating a central black cross wich will destoy the vertical and
// horizontal frequencies
cv::Mat apply_remove_filter(const cv::Mat & img)
{
  cv::Mat shift_complex = img;

  cv::rectangle(
    shift_complex, cv::Point2d(img.cols / 2 - filter_size / 2, 0),
    cv::Point2d(img.cols / 2 + filter_size / 2, img.rows), cv::Scalar(0.0f, 0.0f, 0.0f), -1);
  cv::rectangle(
    shift_complex, cv::Point2d(0, img.rows / 2 - filter_size / 2),
    cv::Point2d(img.cols, img.rows / 2 + filter_size / 2), cv::Scalar(0.0f, 0.0f, 0.0f), -1);

  return shift_complex;
}

cv::Mat apply_filter(const cv::Mat & img, std::string filter_name)
{
  cv::Mat rearrange;

  // Compute the Discrete fourier transform
  cv::Mat complexImg = computeDFT(img);

  // Rearrange quadrants - Spectrum with low values at center - Theory mode
  cv::Mat shift_complex = fftShift(complexImg);

  if (filter_name == "keep_vertical_and_horizontal_frec") {
    shift_complex = apply_keep_filter(shift_complex);

    // Rearrange quadrants - Spectrum with low
    // values at corners - OpenCV mode
    rearrange = fftShift(shift_complex);

    // Get the spectrum after the processing put it in a global variable for display
    spectrum_filter_1 = spectrum(rearrange);

  } else if (filter_name == "remove_vertical_and_horizontal_frec") {
    shift_complex = apply_remove_filter(shift_complex);

    // Rearrange quadrants - Spectrum with low
    // values at corners - OpenCV mode
    rearrange = fftShift(shift_complex);

    // Get the spectrum after the processing and save it in a global variable for display
    spectrum_filter_2 = spectrum(rearrange);
  }

  cv::Mat inverse_transform;
  cv::idft(rearrange, inverse_transform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(inverse_transform, inverse_transform, 0, 1, cv::NORM_MINMAX);
  return inverse_transform;
}

cv::Mat apply_umbral(const cv::Mat in_image, float threshold_p)
{
  cv::Mat img(in_image.rows, in_image.cols, CV_8UC1);
  // Read pixel values
  for (int i = 0; i < in_image.rows; i++) {
    for (int j = 0; j < in_image.cols; j++) {
      // You can now access the pixel value and calculate the new value
      float value = (float)in_image.at<float>(i, j);
      if (value > threshold_p) {
        img.at<uchar>(i, j) = 255;
      } else {
        img.at<uchar>(i, j) = 0;
      }
    }
  }
  return img;
}

void read_key_pressed()
{
  int key_pressed;
  // wait new image format during 100ms
  key_pressed = cv::waitKey(100);

  // change the image format if it is necessary
  if (key_pressed != NO_NEW_FORMAT) {
    if (CAMERA_MODE == UMBRAL && threshold_img_showed == false &&
      key_pressed == DISPLAY_FILTERS)
    {
      cv::imshow("keep_filter_threshold", umbralized_1);
      cv::imshow("remove_filter_threshold", umbralized_2);
      cv::imshow("keep_filter", spectrum_filter_1);
      cv::imshow("remove_filte", spectrum_filter_2);
      threshold_img_showed = true;

    } else if (
      CAMERA_MODE == UMBRAL && threshold_img_showed == true &&
      key_pressed == DISPLAY_FILTERS)
    {
      cv::destroyAllWindows();
      threshold_img_showed = false;

    } else {
      CAMERA_MODE = key_pressed;
    }
  }
}

cv::Mat write_user_text_at_img(cv::Mat out_image)
{
  cv::String text_1 = "1: Gray, 2: Fourier, 3: Keep Filter, 4: Remove Filter, 5: AND";
  cv::String text_2 = "[z,x]: -+ filter vol: 50";

  cv::cvtColor(out_image, out_image, cv::COLOR_GRAY2BGR);

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

cv::Mat image_processing(const cv::Mat in_image)
{
  cv::Mat noise_filtered_img, out_image, Dft, gray_img, umbral_1, umbral_2;

  if (!initialized_img) {
    initialize_user_interface();
  }

  cv::medianBlur(in_image, noise_filtered_img, 3);

  switch (CAMERA_MODE) {
    case FOURIER_SPECTRUM:
      cv::cvtColor(noise_filtered_img, gray_img, cv::COLOR_BGR2GRAY);
      Dft = computeDFT(gray_img);
      out_image = spectrum(Dft);
      break;

    case JUST_VERTICAL_AND_HORIZONTAL_FREQ:
      cv::cvtColor(noise_filtered_img, gray_img, cv::COLOR_BGR2GRAY);
      out_image = apply_filter(gray_img, "keep_vertical_and_horizontal_frec");
      break;

    case NOT_VERTICAL_AND_HORIZONTAL_FREQ:
      cv::cvtColor(noise_filtered_img, gray_img, cv::COLOR_BGR2GRAY);
      out_image = apply_filter(gray_img, "remove_vertical_and_horizontal_frec");
      break;

    case UMBRAL:
      cv::cvtColor(noise_filtered_img, gray_img, cv::COLOR_BGR2GRAY);
      umbral_1 = apply_filter(gray_img, "keep_vertical_and_horizontal_frec");
      umbralized_1 = apply_umbral(umbral_1, 0.6f);
      umbral_2 = apply_filter(gray_img, "remove_vertical_and_horizontal_frec");
      umbralized_2 = apply_umbral(umbral_2, 0.4f);
      cv::bitwise_and(umbralized_1, umbralized_2, out_image);
      break;

    case FILTER_GROW:
      if (filter_size < 100) {
        filter_size += 1.0f;
      }
      cv::cvtColor(noise_filtered_img, out_image, cv::COLOR_BGR2GRAY);
      threshold_img_showed = false;
      CAMERA_MODE = GRAY;

      std::cout << "filter size: " << filter_size << std::endl;
      break;

    case FILTER_REDUCE:
      if (filter_size > 50) {
        filter_size -= 1.0f;
      }
      cv::cvtColor(noise_filtered_img, out_image, cv::COLOR_BGR2GRAY);
      threshold_img_showed = false;
      CAMERA_MODE = GRAY;

      std::cout << "filter size: " << filter_size << std::endl;
      break;

    case GRAY:
    default:
      cv::cvtColor(noise_filtered_img, out_image, cv::COLOR_BGR2GRAY);
      break;
  }

  cv::Mat final_img = write_user_text_at_img(out_image);

  // show resultant image at window P2
  cv::imshow(p2_window_name, final_img);

  read_key_pressed();

  return final_img;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}
