#ifndef CELL_H_
#define CELL_H_
// std
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_map>
// eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "liw/utility.h"

#include <gp3d/gp_types.h>

#include <common/timer/timer.h>
#include "gp3d/pointMatrix.h"

namespace GSLIVM {

enum Direction { X = 0, Y, Z, Unknown };

class Region {
  // define a voxel region, which is cells' borders
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double x_min{0}, y_min{0}, z_min{0}, x_max{0}, y_max{0}, z_max{0};

  Region() {}

  Region(double x_min_, double y_min_, double z_min_, double x_max_, double y_max_, double z_max_)
      : x_min(x_min_), y_min(y_min_), x_max(x_max_), y_max(y_max_), z_max(z_max_), z_min(z_min_) {}
};

class Cell {
  // define a basic data structure of the map, a voxel cell containing points, several layers of mesh, and so on.
 public:
  bool is_surface = false;
  int direction = -1;
  Cell(GpParameter& params_, camOptions& cam_params_, PointMatrix& raw_points, Region region_);

 private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Region region;

  GpParameter gp_options_;
  camOptions cam_params_;

  PointMatrix cell_raw_points;
};
}  // namespace GSLIVM

#endif
