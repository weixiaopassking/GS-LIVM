#pragma once
// c++
#include <math.h>
#include <iostream>

// ros
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

// eigen
#include <Eigen/Core>

// opencv
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "liw/lioOptimization.h"
#include "liw/opticalFlowTracker.h"
#include "liw/rgbMapTracker.h"

#include "liw/cudaMatrixMultiply.h"

#define INIT_COV (0.0001)

class cloudFrame;
class opticalFlowTracker;
class rgbMapTracker;

class imageProcessing {
 private:
  int image_width;   // raw image width
  int image_height;  // raw image height

  Eigen::Matrix3d camera_intrinsic;
  Eigen::Matrix<double, 5, 1> camera_dist_coeffs;

  cv::Mat intrinsic, dist_coeffs;
  cv::Mat m_ud_map1, m_ud_map2;

  Eigen::Matrix3d R_imu_camera;
  Eigen::Vector3d t_imu_camera;

  bool ifEstimateCameraIntrinsic;
  bool ifEstimateExtrinsic;

  Eigen::Matrix<double, 11, 11> covariance;

  double image_resize_ratio;

  bool log_time = false;
  bool first_data;

  int maximum_tracked_points;

  int track_windows_size;

  int num_iterations;

  double tracker_minimum_depth;
  double tracker_maximum_depth;

  double cam_measurement_weight;
  CudaMatrixMultiply cublasMul;

 public:
  double time_last_process;

  opticalFlowTracker* op_tracker;
  rgbMapTracker* map_tracker;

  imageProcessing();

  void initCameraParams();

  void setImageWidth(int& para);

  void setImageRatio(double& para);

  int getImageWidth() { return image_width; };

  void setImageHeight(int& para);

  int getImageHeight() { return image_height; };

  int getfx() { return camera_intrinsic(0, 0); }

  int getfy() { return camera_intrinsic(1, 1); }

  int getcx() { return camera_intrinsic(0, 2); }

  void setLogtime(bool& para) { log_time = para; };

  int getcy() { return camera_intrinsic(1, 2); }

  void setCameraIntrinsic(std::vector<double>& v_camera_intrinsic);
  void setCameraDistCoeffs(std::vector<double>& v_camera_dist_coeffs);

  void setExtrinR(Eigen::Matrix3d& R);
  void setExtrinT(Eigen::Vector3d& t);

  void setInitialCov();

  Eigen::Matrix3d getCameraIntrinsic();

  bool process(voxelHashMap& voxel_map, cloudFrame* p_frame);

  void imageEqualize(cv::Mat& image, int amp);
  cv::Mat initCubicInterpolation(cv::Mat& image);
  cv::Mat equalizeColorImageYcrcb(cv::Mat& image);

  bool vioEsikf(cloudFrame* p_frame);

  bool vioPhotometric(cloudFrame* p_frame);

  void updateCameraParameters(cloudFrame* p_frame, Eigen::Matrix<double, 11, 1>& d_x);

  void updateCameraParameters(cloudFrame* p_frame, Eigen::Matrix<double, 6, 1>& d_x);

  void printParameter();
};