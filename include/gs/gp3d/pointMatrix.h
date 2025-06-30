#pragma once
#ifndef POINTMATRIX_H_
#define POINTMATRIX_H_
// eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <iostream>
#include <map>

namespace GSLIVM {

class PointMatrix {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int num_point = 0;
  const int resize_step = 8;  // when size of point matrix is not enough, resize it
  const int init_size = 8;    // 512 256

  Eigen::Matrix<double, 3, Eigen::Dynamic> point;
  Eigen::Matrix<double, 1, Eigen::Dynamic> variance;

  Eigen::Matrix<double, 3, 1> centroid{0, 0, 0};
  Eigen::Matrix3d eigenvectorMat;
  Eigen::Matrix3d eigenvalMat;
  std::map<double, Eigen::Matrix<double, 3, 1>> eig_sorted;

  PointMatrix() {
    // initialize
    variance = Eigen::MatrixXd::Zero(1, init_size);
    point = Eigen::MatrixXd::Zero(3, init_size);
    variance.fill(100);  // 100 is a very large value, means not usable
  }

  void clear_quick() {
    // clear data by changing num_point, save time
    num_point = 0;
    centroid.fill(0);
    eigenvectorMat.fill(0);
    eigenvalMat.fill(0);
    eig_sorted.clear();
  }

  void clear() {
    // clear all data
    point.fill(0);
    variance.fill(100);
    clear_quick();
  }

  PointMatrix& operator=(const Eigen::MatrixXd& points_copy) {
    // use Eigen::MatrixXd to initialize
    if (points_copy.cols() > point.cols()) {
      point.resize(3, points_copy.cols());
      variance.resize(1, points_copy.cols());
    }
    clear();

    point.leftCols(points_copy.cols()) = points_copy.topRows(3);

    num_point = int(points_copy.cols());
    if (points_copy.rows() == 4) {
      variance.leftCols(points_copy.cols()) = points_copy.row(3);
    } else {
      variance.leftCols(points_copy.cols()).fill(100);
    }
    return *this;
  }

  void addPoint(const Eigen::Matrix<double, 3, 1>& point_add, float variance_sensor) {
    // add a point
    if (num_point + 1 > point.cols()) {
      point.conservativeResize(Eigen::NoChange_t(3), point.cols() + resize_step);
      variance.conservativeResize(Eigen::NoChange_t(1), point.cols() + resize_step);

      point.rightCols(resize_step).fill(0);
      variance.rightCols(resize_step).fill(100);
    }

    point.col(num_point) = point_add;
    variance(0, num_point) = variance_sensor;

    num_point++;
  }

  void sortEigenPairs(Eigen::Matrix3d& eigenvectorMat, Eigen::Matrix3d& eigenvalMat) {
    std::vector<std::pair<Eigen::Vector3d, double>> eigenPairs;
    for (int i = 0; i < 3; ++i) {
      eigenPairs.push_back(std::make_pair(eigenvectorMat.col(i), eigenvalMat(i, 0)));
    }

    std::sort(
        eigenPairs.begin(),
        eigenPairs.end(),
        [](const std::pair<Eigen::Vector3d, double>& a, const std::pair<Eigen::Vector3d, double>& b) {
          return a.second < b.second;
        });

    for (int i = 0; i < 3; ++i) {
      eigenvectorMat.col(i) = eigenPairs[i].first;
      eigenvalMat(i, 0) = eigenPairs[i].second;
    }
  }

  void eigenDecomposition() {
    // Eigen decomposition
    if (num_point != 0) {
      centroid = point.leftCols(num_point).rowwise().sum() / num_point;

      Eigen::Matrix<double, 3, Eigen::Dynamic> remove_centroid =
          point.leftCols(num_point) - centroid.replicate(1, num_point);
      Eigen::Matrix3d covarianceMat = remove_centroid * remove_centroid.adjoint() / num_point;
      Eigen::EigenSolver<Eigen::Matrix3d> eig(covarianceMat);
      eigenvalMat = eig.pseudoEigenvalueMatrix();
      eigenvectorMat = eig.pseudoEigenvectors();

      for (int i = 0; i < 3; i++) {
        eig_sorted.emplace(eigenvalMat(i, i), eigenvectorMat.col(i));
      }

      sortEigenPairs(eigenvalMat, eigenvectorMat);
    } else {
      std::cout << "eigenDecomposition error: no point" << std::endl;
    }
  }
};
}  // namespace GSLIVM
#endif
