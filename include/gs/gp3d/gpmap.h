#ifndef MAP_H
#define MAP_H

#include <omp.h>
#include "gp3d/gpcell.h"

namespace GSLIVM {
struct Vector3DHasher {
  std::size_t operator()(const Eigen::Vector3d& vec) const {
    const size_t kP1 = 73856093;
    const size_t kP2 = 19349669;
    const size_t kP3 = 83492791;
    return vec.x() * kP1 + vec.y() * kP2 + vec.z() * kP3;
  }
};

struct varianceUpdate {
  std::size_t hash_;
  int num_point;
  std::vector<double> update_variance;
};

struct needGPUdata {
  std::size_t hash_;
  Eigen::Matrix<double, 3, Eigen::Dynamic> point;
  Eigen::Matrix<double, 1, Eigen::Dynamic> variance;
  int num_point;
  Region region_;
  int direction_;
  bool is_converged{false};
  std::vector<Eigen::VectorXd> gmm_init_mu;
  std::vector<Eigen::MatrixXd> gmm_init_sigma;
};

class VoxelNode {
 public:
  std::size_t hash;
  bool is_converged{false};

 public:
  VoxelNode(std::size_t& hash_) : hash{hash_} {};
};

class GpMap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  PointMatrix points_notadded;
  PointMatrix points_scan;

  std::unordered_map<std::size_t, PointMatrix> hash_pointmatrix;
  std::unordered_map<std::size_t, VoxelNode> hash_voxelnode;
  std::unordered_map<std::size_t, Eigen::Vector3d> hash_vecpoint;

  std::vector<std::size_t> updated_voxel;
  GpParameter gp_options_;
  camOptions cam_params_;

 public:
  GpMap(GpParameter& params_);

  void dividePointsIntoCellInitMap(
      GSLIVM::ImageRt& frame,
      bool& is_init,
      Eigen::Matrix<double, 3, Eigen::Dynamic>& points_curr,
      std::vector<GSLIVM::needGPUdata>& updataMapData,
      std::vector<GSLIVM::GsForLoss>& frameLossPoints);

  void updateVariance(std::vector<varianceUpdate>& update_vas_);
  void splitPointsIntoCell(PointMatrix& points, std::vector<GSLIVM::GsForLoss>& frameLossPoints);
};
}  // namespace GSLIVM
#endif  // SLAMESH_MAP_H
