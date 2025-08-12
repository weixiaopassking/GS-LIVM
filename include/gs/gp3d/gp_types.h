#ifndef GP_TYPES_H__
#define GP_TYPES_H__

#include <queue>
#include <vector>

// opencv
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#define MAX_SIMI 500

namespace GSLIVM {
struct ImageRt {
  cv::Mat image;
  Eigen::Quaterniond R;
  Eigen::Vector3d t;
  Eigen::Matrix3d R_imu_lidar;
  Eigen::Vector3d t_imu_lidar;
  Eigen::Matrix3d R_camera_lidar;
  Eigen::Vector3d t_camera_lidar;
};

struct GsForMap {
  std::size_t hash_posi_;
  torch::Tensor gs_xyz;
  torch::Tensor gs_rgb;
  torch::Tensor gs_cov;
};

struct GsForMaps {
  std::vector<std::size_t> hash_posi_s;
  std::vector<std::vector<int>> indexes;
  torch::Tensor gs_xyzs;
  torch::Tensor gs_rgbs;
  torch::Tensor gs_covs;
};

struct GsForLoss {
  std::size_t hash_posi_;
  torch::Tensor gs_xyz;
};

struct GsForLosses {
  std::unordered_map<std::size_t, torch::Tensor> _losses;
};

struct DeltaSimi {
  torch::Tensor depth;
  torch::Tensor depth_sol;
  Eigen::Matrix3f cam_pose_R;
  Eigen::Vector3f cam_pose_t;
  Eigen::Matrix3f K;
  Eigen::Matrix3f inv_K;
};

struct camOptions {
  int cam_height;
  int cam_width;
  int channels;
  double fx;
  double fy;
  double cx;
  double cy;

  double d0;
  double d1;
  double d2;
  double d3;

  double blind_far;
};

class GpParameter {
 public:
  int min_points_num_to_gp, num_gp_side, neighbour_size, curr_cam_per_iter, history_cam_per_iter;

  int image_sliding_window;
  double variance_sensor, grid, kernel_size;

  double eigen_1, max_var_mean;  // PCA

  bool full_cover;
  bool debug;
  bool log_time;
  GpParameter() {};
};

}  // namespace GSLIVM

#endif
