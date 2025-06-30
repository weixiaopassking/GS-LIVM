#include "gp3d/gpcell.h"

namespace GSLIVM {

Cell::Cell(GpParameter& params_, camOptions& cam_params_, PointMatrix& raw_points, Region region_)
    : gp_options_{params_}, cam_params_{cam_params_}, cell_raw_points(raw_points), region(region_) {

  // use gaussian process to reconstruct the local surfaces inside a cell, one cell can have 3 surfaces in 3 directions
  // decide direction of GP, based on PCA (Principal Component Analysis)
  Eigen::Matrix<double, 1, 3> angle_a;
  std::map<double, int> sorted_angle_a;

  cell_raw_points.eigenDecomposition();

  angle_a = (cell_raw_points.eig_sorted.begin()->second.transpose() * Eigen::MatrixXd::Identity(3, 3)).array().acos();

  // sort
  for (int i = 0; i < 3; i++) {
    angle_a(0, i) = (angle_a(0, i) > M_PI / 2) ? M_PI - angle_a(0, i) : angle_a(0, i);
    sorted_angle_a.emplace(angle_a(0, i), i);
  }

  if (cell_raw_points.eig_sorted.rbegin()->first / (++cell_raw_points.eig_sorted.begin())->first >
      gp_options_.eigen_1) {
    is_surface = true;
    direction = sorted_angle_a.begin()->second;
  } else {
    is_surface = false;
  }
};

}  // namespace GSLIVM