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
#
# Autor: Adrián Cobo Merino
# Partes implementadas:
# - Detección de pelota en 2D y proyección 3D
# - Detección de pelota en 3D y proyección 2D
# - Proyección líneas
# - Funcionalidad extra:
#   - Proyección de la pelota de 3D a 2D teniendo en cuenta el radio calculado en 3D y
#     dibujarlo sobre la imagen (medio).
#   - Proyección de la pelota de 2D a 3D teniendo en cuenta el radio calculado en 2D de
#     manera que el centro se proyecte donde debería y no en la cara frontal (medio).
#
# ¡¡¡ Importante !!!
# Antes de ejecutar el programa, copiar los ficheros de configuracion de yolo en la ruta.../your_ws/cfg/, para ello:
# 1 Crea la carptera cfg en el directorio raiz tu ws y ejecuta wget https://pjreddie.com/media/files/yolov3.weights 
# 2 Copia el archivo coco.names en esa misma carpeta. 
*/

#include <image_geometry/pinhole_camera_model.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <time.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <image_transport/image_transport.hpp>
#include <iostream>
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#define TF_HEADFRONT_BASE_X 0.047
#define TF_HEADFRONT_BASE_Y 1.067
#define TF_HEADFRONT_BASE_Z -0.216
#define CLASS_ID_PERSON 0

using namespace std::chrono_literals;

const int CLEAN = 0;     // option 1
const int OPTION_1 = 1;  // option 2
const int OPTION_2 = 2;  // option 3
const int MAX_OPTIONS = 2;
const int MAX_DISTANCE = 8;

const char * proyect_window_name = "PROYECT";
const char * trackbar_1_text = "Option";
const char * trackbar_2_text = "Distance";

int CAMERA_MODE = CLEAN;
int TRACKBAR_DISTANCE;

bool initialized_img = false;
bool person_detected;

const cv::Scalar LOWER_PINK = cv::Scalar(145, 75, 75);
const cv::Scalar UPPER_PINK = cv::Scalar(165, 255, 255);

cv::Mat depth_image;  // normal depth img
image_geometry::PinholeCameraModel intrinsic_camera_matrix;
geometry_msgs::msg::TransformStamped camera2basefootprint;

std::vector<cv::Point> balls_center_2d;
std::vector<pcl::PointXYZRGB> balls_center_3d;
std::vector<int> radius_2d;
std::vector<float> radius_3d;

// Initialize the parameters
float confThreshold = 0.5;  // Confidence threshold
float nmsThreshold = 0.4;   // Non-maximum suppression threshold
int inpWidth = 640;         // Width of network's input image
int inpHeight = 480;        // Height of network's input image
std::vector<std::string> classes;
cv::dnn::Net net;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat & frame, const std::vector<cv::Mat> & out);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net & net);

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);
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

class PCLSubscriber : public rclcpp::Node
{
public:
  PCLSubscriber()
  : Node("pcl_subscriber")
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

void initialize_yolo_interface()
{
  // Load names of classes
  std::string classesFile = "cfg/coco.names";
  std::ifstream ifs(classesFile.c_str());
  std::string line;
  while (getline(ifs, line)) {
    classes.push_back(line);
  }

  std::string device = "gpu";
  // Give the configuration and weight files for the model
  cv::String modelConfiguration = "cfg/yolov3.cfg";
  cv::String modelWeights = "cfg/yolov3.weights";

  // Load the network
  net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

  if (device == "cpu") {
    std::cout << "Using CPU device" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
  } else if (device == "gpu") {
    std::cout << "Using GPU device" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  }
}

void initialize_user_interface()
{
  cv::namedWindow(proyect_window_name, 0);

  // create Trackbar and add to a window
  cv::createTrackbar(trackbar_1_text, proyect_window_name, nullptr, MAX_OPTIONS, 0);
  cv::createTrackbar(trackbar_2_text, proyect_window_name, nullptr, MAX_DISTANCE, 0);

  // set Trackbar’s value
  cv::setTrackbarPos(trackbar_1_text, proyect_window_name, 0);
  cv::setTrackbarPos(trackbar_2_text, proyect_window_name, 0);
  initialize_yolo_interface();

  initialized_img = true;
}

cv::Mat color_filter(cv::Mat src, cv::Scalar lower_range_color, cv::Scalar uper_range_color)
{
  cv::Mat hsv, mask_green, image_color_filtered;

  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  cv::Mat color_mask;
  cv::inRange(hsv, lower_range_color, uper_range_color, color_mask);

  cv::bitwise_and(hsv, hsv, image_color_filtered, color_mask);

  return image_color_filtered;
}

cv::Point proyect_3d_to_2d(double px, double py, double pz, cv::Mat rot_tras_mat)
{
  uint x_2d, y_2d;

  cv::Mat p_3d = (cv::Mat_<double>(4, 1) << px, py, pz, 1.0);
  cv::Mat p_2d = intrinsic_camera_matrix.intrinsicMatrix() * (rot_tras_mat * p_3d);

  // normalize the 2d pixel coordinates
  x_2d = (uint)(p_2d.at<double>(0, 0) / p_2d.at<double>(2, 0));
  y_2d = (uint)(p_2d.at<double>(1, 0) / p_2d.at<double>(2, 0));

  return cv::Point(x_2d, y_2d);
}

cv::Mat print_detected_circles(
  const cv::Mat src, const std::vector<cv::Vec3f> circles,
  bool use_radius)
{
  cv::Mat img_circles_printed = src.clone();
  balls_center_2d.clear();
  radius_2d.clear();

  // create ou rototraslation matrix with no rotation in this case because we ae goint to use the
  // refence axis from camera frame instead of the base_footprint frame coordinates
  cv::Mat rot_tras_mat =
    (cv::Mat_<double>(3, 4) << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

  for (size_t i = 0; i < circles.size(); i++) {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    balls_center_2d.push_back(center);
    radius_2d.push_back((int)c[2]);

    if (!use_radius) {
      // circle center
      cv::circle(img_circles_printed, center, 1, cv::Scalar(150, 150, 150), 3, cv::LINE_AA);
      // circle outline
      int radius = c[2];
      cv::circle(img_circles_printed, center, radius, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }
  }

  if (use_radius) { // extra option draw 2d circle using 3d detection radius and center
    for (size_t i = 0; i < balls_center_3d.size(); i++) {
      pcl::PointXYZRGB c = balls_center_3d[i];
      cv::Point center = proyect_3d_to_2d((double)c.x, (double)c.y, (double)c.z, rot_tras_mat);
      cv::Point auxiliar = proyect_3d_to_2d(
        (double)c.x - (double)radius_3d[i], (double)c.y, (double)c.z, rot_tras_mat);
      uint radius = abs(center.x - auxiliar.x) + abs(center.y - auxiliar.y);
      // circle center
      cv::circle(img_circles_printed, center, radius, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    }

    return img_circles_printed;
  }

  for (size_t i = 0; i < balls_center_3d.size(); i++) {
    pcl::PointXYZRGB c = balls_center_3d[i];
    cv::Point center = proyect_3d_to_2d((double)c.x, (double)c.y, (double)c.z, rot_tras_mat);

    // circle center
    cv::circle(img_circles_printed, center, 1, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
  }

  return img_circles_printed;
}

cv::Mat detect_pink_balls(const cv::Mat src, bool use_radius)
{
  cv::Mat blue_filtered, auxiliar_bgr, auxiliar_gray, out_image;
  std::vector<cv::Vec3f> circles;

  blue_filtered = color_filter(src, LOWER_PINK, UPPER_PINK);

  cv::medianBlur(blue_filtered, blue_filtered, 9);
  cv::cvtColor(blue_filtered, auxiliar_bgr, cv::COLOR_HSV2BGR);
  cv::cvtColor(auxiliar_bgr, auxiliar_gray, cv::COLOR_BGR2GRAY);

  // change 5th argumentto detect circles with different distances to each other
  // change the last two parameters (min_radius & max_radius) to detect larger circles
  cv::HoughCircles(
    auxiliar_gray, circles, cv::HOUGH_GRADIENT, 1, auxiliar_gray.rows / 16, 160.0, 12.0, 15,
    77);

  return out_image = print_detected_circles(src, circles, use_radius);
}

cv::Mat print_3d_to_2d_proyection_lines(cv::Mat input)
{
  double x, y, z;
  x = camera2basefootprint.transform.translation.x;
  y = camera2basefootprint.transform.translation.y;
  z = camera2basefootprint.transform.translation.z;

  // create ou rototraslation matrix with no rotation in this case because we ae goint to use the
  // refence axis from camera frame instead of the base_footprint frame coordinates
  cv::Mat rot_tras_mat =
    (cv::Mat_<double>(3, 4) << 1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z);

  for (int i = 0; i <= TRACKBAR_DISTANCE; i++) {
    cv::Point p1 = proyect_3d_to_2d(1, 0.0, double(i), rot_tras_mat);
    cv::Point p2 = proyect_3d_to_2d(-1, 0.0, double(i), rot_tras_mat);

    cv::line(input, p1, p2, cv::Scalar(80, 30 * i, 80), 2);
    p1.x = p1.x + 10;
    cv::putText(
      input, std::to_string(i), p1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 30 * i, 80),
      2);
  }

  return input;
}

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net & net)
{
  static std::vector<cv::String> names;
  if (names.empty()) {
    // Get the indices of the output layers, i.e. the layers with unconnected outputs
    std::vector<int> outLayers = net.getUnconnectedOutLayers();

    // get the names of all the layers in the network
    std::vector<cv::String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
    }
  }
  return names;
}

// Remove the bounding boxes with low confidence
std::vector<int> postprocess(const std::vector<cv::Mat> & outs)
{
  std::vector<int> classIds;

  // Scan through all the bounding boxes output from the network and keep only the
  // ones with high confidence scores.
  for (size_t i = 0; i < outs.size(); ++i) {
    float * data = (float *)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      cv::Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold) {
        classIds.push_back(classIdPoint.x);
      }
    }
  }

  return classIds;
}

bool detect_person(cv::Mat frame)
{
  // Create a window
  cv::Mat blob;

  // Create a 4D blob from a frame.
  cv::dnn::blobFromImage(
    frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);

  // Sets the input to the network
  net.setInput(blob);

  // Runs the forward pass to get output of the output layers
  std::vector<cv::Mat> outs;
  net.forward(outs, getOutputsNames(net));

  // Remove the bounding boxes with low confidence
  std::vector<int> classIds = postprocess(outs);

  // Search for person detection
  for (size_t i = 0; i < classIds.size(); i++) {
    if (classIds[i] == CLASS_ID_PERSON) {
      return true;
    }
  }
  return false;
}

cv::Mat image_processing(const cv::Mat in_image)
{
  cv::Mat noise_filtered_img, detected_balls, proyection_lines_printed, out_image;

  if (!initialized_img) {
    initialize_user_interface();
  }

  // refresh the global paramiters
  CAMERA_MODE = cv::getTrackbarPos(trackbar_1_text, proyect_window_name);
  TRACKBAR_DISTANCE = cv::getTrackbarPos(trackbar_2_text, proyect_window_name);

  cv::medianBlur(in_image, noise_filtered_img, 3);

  person_detected = detect_person(noise_filtered_img);

  switch (CAMERA_MODE) {
    case OPTION_1:
      if (person_detected) {
        detected_balls = detect_pink_balls(noise_filtered_img, false);
        out_image = print_3d_to_2d_proyection_lines(detected_balls);
      } else {
        detect_pink_balls(noise_filtered_img, false);
        out_image = in_image;
      }
      break;

    case OPTION_2:
      out_image = detect_pink_balls(noise_filtered_img, true);
      break;

    case CLEAN:
    default:
      out_image = in_image;
      break;
  }

  // show resultant image at window P3
  cv::imshow(proyect_window_name, out_image);
  cv::waitKey(100);

  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  // cv::cvtColor(out_image, out_image, cv::COLOR_GRAY2BGR);
  return out_image;
}

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
  pcl::PointCloud<pcl::PointXYZRGB> dst_pointcloud, pcl::PointXYZRGB center, double size, int r,
  int g, int b)
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
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;    // Create the filtering object
  float x, y, z, r;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr spheres_point_cloud(in_pointcloud.makeShared());
  int nr_points = (int)spheres_point_cloud->size();
  seg = initialice_segmentator();

  balls_center_3d.clear();
  radius_3d.clear();

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
    r = coefficients->values[3];

    pcl::PointXYZRGB p = pcl::PointXYZRGB(x, y, z, 0, 0, 0);
    balls_center_3d.push_back(p);
    radius_3d.push_back(r);
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
  double px, py, pz;

  for (int i = 3; i <= TRACKBAR_DISTANCE; i++) {
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

pcl::PointCloud<pcl::PointXYZRGB> centers_2d_to_3d(
  pcl::PointCloud<pcl::PointXYZRGB> input,
  bool use_radius)
{
  double fx, fy, cx, cy, x_3d, y_3d, z_3d, d, x2_3d;
  int px, py;

  cv::Mat intrinsic_marix = (cv::Mat)intrinsic_camera_matrix.intrinsicMatrix();

  fx = intrinsic_marix.at<double>(0, 0);
  fy = intrinsic_marix.at<double>(1, 1);
  cx = intrinsic_marix.at<double>(0, 2);
  cy = intrinsic_marix.at<double>(1, 2);

  for (size_t i = 0; i < balls_center_2d.size(); i++) {
    px = balls_center_2d[i].x;
    py = balls_center_2d[i].y;
    d = depth_image.at<float>(py, px);

    x_3d = ((double)px - cx) * d / fx;
    y_3d = ((double)py - cy) * d / fy;

    if (use_radius) { // extra option draw a 3d cube at the cencer of the sphere using 2d detection radius and center
      x2_3d = ((double)px + (double)radius_2d[i] - cx) * d / fx;
      z_3d = d + abs(x_3d - x2_3d);
    } else {
      z_3d = d;
    }

    pcl::PointXYZRGB p = pcl::PointXYZRGB(x_3d, y_3d, z_3d, 100, 100, 100);
    input = draw_cube(input, p, 0.1, 0, 0, 0);
  }

  return input;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> pink_points, spheres, out_pointcloud;

  switch (CAMERA_MODE) {
    case OPTION_1:
      if (person_detected) {
        pink_points = ball_filter(in_pointcloud);
        spheres = pcl_sphere(pink_points);
        spheres = centers_2d_to_3d(spheres, false);
        out_pointcloud = draw_floor_cubes(spheres);
      } else {
        out_pointcloud = in_pointcloud;
      }
      break;

    case OPTION_2:
      pink_points = ball_filter(in_pointcloud);
      spheres = pcl_sphere(pink_points);
      out_pointcloud = centers_2d_to_3d(spheres, true);
      break;

    case CLEAN:
    default:
      out_pointcloud = in_pointcloud;
      break;
  }

  return out_pointcloud;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::SingleThreadedExecutor exec;

  auto cv_node = std::make_shared<ComputerVisionSubscriber>();
  auto pcl_node = std::make_shared<PCLSubscriber>();
  exec.add_node(cv_node);
  exec.add_node(pcl_node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
