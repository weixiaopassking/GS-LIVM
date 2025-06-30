#pragma once

// c++
#include <math.h>
#include <iostream>

// eigen
#include <Eigen/Core>

// opencv
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "liw/lioOptimization.h"
#include "liw/lkpyramid.h"
#include "liw/utility.h"

#ifdef EMPTY_ANSI_COLORS
#define ANSI_COLOR_RED ""
#define ANSI_COLOR_RED_BOLD ""
#define ANSI_COLOR_GREEN ""
#define ANSI_COLOR_GREEN_BOLD ""
#define ANSI_COLOR_YELLOW ""
#define ANSI_COLOR_YELLOW_BOLD ""
#define ANSI_COLOR_BLUE ""
#define ANSI_COLOR_BLUE_BOLD ""
#define ANSI_COLOR_MAGENTA ""
#define ANSI_COLOR_MAGENTA_BOLD ""
#else
#define ANSI_COLOR_RED "\x1b[0;31m"
#define ANSI_COLOR_RED_BOLD "\x1b[1;31m"
#define ANSI_COLOR_RED_BG "\x1b[0;41m"

#define ANSI_COLOR_GREEN "\x1b[0;32m"
#define ANSI_COLOR_GREEN_BOLD "\x1b[1;32m"
#define ANSI_COLOR_GREEN_BG "\x1b[0;42m"

#define ANSI_COLOR_YELLOW "\x1b[0;33m"
#define ANSI_COLOR_YELLOW_BOLD "\x1b[1;33m"
#define ANSI_COLOR_YELLOW_BG "\x1b[0;43m"

#define ANSI_COLOR_BLUE "\x1b[0;34m"
#define ANSI_COLOR_BLUE_BOLD "\x1b[1;34m"
#define ANSI_COLOR_BLUE_BG "\x1b[0;44m"

#define ANSI_COLOR_MAGENTA "\x1b[0;35m"
#define ANSI_COLOR_MAGENTA_BOLD "\x1b[1;35m"
#define ANSI_COLOR_MAGENTA_BG "\x1b[0;45m"

#define ANSI_COLOR_CYAN "\x1b[0;36m"
#define ANSI_COLOR_CYAN_BOLD "\x1b[1;36m"
#define ANSI_COLOR_CYAN_BG "\x1b[0;46m"

#define ANSI_COLOR_WHITE "\x1b[0;37m"
#define ANSI_COLOR_WHITE_BOLD "\x1b[1;37m"
#define ANSI_COLOR_WHITE_BG "\x1b[0;47m"

#define ANSI_COLOR_BLACK "\x1b[0;30m"
#define ANSI_COLOR_BLACK_BOLD "\x1b[1;30m"
#define ANSI_COLOR_BLACK_BG "\x1b[0;40m"

#define ANSI_COLOR_RESET "\x1b[0m"

#define ANSI_DELETE_LAST_LINE "\033[A\33[2K\r"
#define ANSI_DELETE_CURRENT_LINE "\33[2K\r"
#define ANSI_SCREEN_FLUSH std::fflush(stdout);

#define SET_PRINT_COLOR(a) cout << a;

#endif

struct _Scope_color {
  _Scope_color(const char* color) { std::cout << color; }

  ~_Scope_color() { std::cout << ANSI_COLOR_RESET; }
};

#define scope_color(a) _Scope_color _scope(a);

class cloudFrame;
class rgbMapTracker;

class opticalFlowTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  cv::Mat old_image, old_gray;
  cv::Mat cur_image, cur_gray;
  cv::Mat cur_mask;

  unsigned int image_idx = 0;
  double last_image_time, current_image_time;

  std::vector<int> cur_ids, old_ids;
  unsigned int maximum_tracked_points;

  //   cv::Mat m_ud_map1, m_ud_map2;
  //   cv::Mat dist_coeffs;
  cv::Mat intrinsic;

  std::vector<cv::Point2f> last_tracked_points, cur_tracked_points;

  std::vector<void*> rgb_points_ptr_vec_in_last_image;
  std::map<void*, cv::Point2f> map_rgb_points_in_last_image_pose;
  std::map<void*, cv::Point2f> map_rgb_points_in_cur_image_pose;

  std::map<int, std::vector<cv::Point2f>> map_id_points_vec;
  std::map<int, std::vector<int>> map_id_points_image;
  std::map<int, std::vector<cv::Point2f>> map_image_points;

  Eigen::Quaterniond last_quat;
  Eigen::Vector3d last_trans;

  std::shared_ptr<LKOpticalFlowKernel> lk_optical_flow_kernel;

  opticalFlowTracker();

  void setIntrinsic(Eigen::Matrix3d intrinsic_, Eigen::Matrix<double, 5, 1> dist_coeffs_, cv::Size image_size_);

  bool trackImage(cloudFrame* p_frame, double distance);

  void init(cloudFrame* p_frame, std::vector<rgbPoint*>& rgb_points_vec, std::vector<cv::Point2f>& points_2d_vec);

  void setTrackPoints(cv::Mat& image, std::vector<rgbPoint*>& rgb_points_vec, std::vector<cv::Point2f>& points_2d_vec);

  template <typename T>
  void reduce_vector(std::vector<T>& v, std::vector<uchar> status) {
    int j = 0;

    for (unsigned int i = 0; i < v.size(); i++)
      if (status[i])
        v[j++] = v[i];

    v.resize(j);
  }

  void updateAndAppendTrackPoints(cloudFrame* p_frame, rgbMapTracker* map_tracker, double mini_distance = 10.0, int minimum_frame_diff = 3e8);

  void rejectErrorTrackingPoints(cloudFrame* p_frame, double dis = 2.0);

  void updateLastTrackingVectorAndIds();

  bool removeOutlierUsingRansacPnp(cloudFrame* p_frame, int if_remove_ourlier = 1);
};