
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <open3d/Open3D.h>

#include "GMMFit.h"

struct GMMPoint {
  float x;
  float y;
  float z;
};

struct GMMColor {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct GMMNormals {
  float nx;
  float ny;
  float nz;
};

struct GMMScale {
  float cov_x;
  float cov_y;
  float cov_z;
};

struct GMMPointCloud {
  std::vector<GMMPoint> _points;
  std::vector<GMMNormals> _normals;
  std::vector<GMMScale> _scales;
  std::vector<GMMColor> _colors;
};

struct DPointGMM {
  float x, y, z, variance, gray;
};

std::vector<DPointGMM> readDPointGMMFromTxt(const std::string& filePath) {
  std::ifstream inFile(filePath);
  if (!inFile.is_open()) {
    std::cerr << "无法打开文件: " << filePath << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<DPointGMM> data;
  DPointGMM point;
  while (inFile >> point.x >> point.y >> point.z >> point.variance >> point.gray) {
    data.push_back(point);
  }

  inFile.close();
  return data;
}

std::vector<std::vector<DPointGMM>> readAllTXTInFolder(const std::string& folderPath) {
  std::vector<std::vector<DPointGMM>> allGMMPoints;
  for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
    if (entry.is_regular_file() && entry.path().extension() == ".txt") {
      std::vector<DPointGMM> gmmPoints = readDPointGMMFromTxt(entry.path());
      allGMMPoints.push_back(gmmPoints);
    }
  }
  return allGMMPoints;
}

Eigen::MatrixXd convertToEigenMatrix(const std::vector<DPointGMM>& vec) {
  Eigen::MatrixXd eigenMatrix(3, vec.size());  // 初始化n*4的矩阵

  int rowIndex = 0;
  for (const auto& point : vec) {
    eigenMatrix(0, rowIndex) = static_cast<double>(point.x);
    eigenMatrix(1, rowIndex) = static_cast<double>(point.y);
    eigenMatrix(2, rowIndex) = static_cast<double>(point.z);
    // eigenMatrix(3, rowIndex) = static_cast<double>(point.gray);
    rowIndex++;
  }

  return eigenMatrix;
}

std::shared_ptr<open3d::geometry::TriangleMesh>
CreateEllipsoid(const Eigen::Vector3d& center, const Eigen::Vector3d& scale_, const Eigen::Vector3d& color, double sigma_multiplier, int resolution) {
  auto mesh = open3d::geometry::TriangleMesh::CreateSphere(1.0);
  mesh->ComputeVertexNormals();

  // Create scaling transformation
  Eigen::Matrix4d scale_matrix = Eigen::Matrix4d::Identity();

  auto scale = scale_.cwiseSqrt();
  scale_matrix(0, 0) = sigma_multiplier * scale(0);
  scale_matrix(1, 1) = sigma_multiplier * scale(1);
  scale_matrix(2, 2) = sigma_multiplier * scale(2);

  // Apply transformations
  mesh->Transform(scale_matrix);                                         // Apply scaling
  mesh->Translate(Eigen::Vector3d(center.x(), center.y(), center.z()));  // Apply translation

  // Set random color
  mesh->PaintUniformColor(color);

  return mesh;
}

// Function to visualize 3D GMM with Open3D in C++
void Visualize3DGMMOpen3D(const Eigen::MatrixXd& gs_before, const GMMPointCloud& gs_after) {
  // Create an Open3D Visualizer object
  open3d::visualization::Visualizer vis;
  vis.CreateVisualizerWindow();

  auto pcd = std::make_shared<open3d::geometry::PointCloud>();
  for (int i = 0; i < gs_before.cols(); i++) {
    pcd->points_.push_back(gs_before.col(i));
    pcd->colors_.push_back(Eigen::Vector3d(0.2, 0.4, 0.1));
  }

  // Add point cloud to the scene
  vis.AddGeometry(pcd);

  int n_gaussians = gs_after._points.size();

  for (int i = 0; i < n_gaussians; ++i) {
    // Create an ellipsoid mesh
    auto mesh = CreateEllipsoid(
        Eigen::Vector3d(gs_after._points[i].x, gs_after._points[i].y, gs_after._points[i].z),
        Eigen::Vector3d(gs_after._scales[i].cov_x, gs_after._scales[i].cov_y, gs_after._scales[i].cov_z),
        Eigen::Vector3d(gs_after._colors[i].r / 255.0, gs_after._colors[i].g / 255.0, gs_after._colors[i].b / 255.0),
        1,
        30);
    vis.AddGeometry(mesh);
  }

  // Start visualization
  vis.Run();
  vis.DestroyVisualizerWindow();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "使用方法: " << argv[0] << " <文件夹路径>" << std::endl;
    return -1;
  }

  std::string folderPath = argv[1];
  std::vector<std::vector<DPointGMM>> allGMMData = readAllTXTInFolder(folderPath);

  std::cout << allGMMData.size() << std::endl;
  int iter = 0;
  for (auto& vec : allGMMData) {

    Eigen::MatrixXd w_X = convertToEigenMatrix(vec);

    if (w_X.cols() == 0) {
      continue;
    }

    int num_gmm = static_cast<int>(std::ceil(w_X.cols() / 100.0));
    // if (num_gmm > 5) {
    //   continue;
    // }
    std::cout << "iter: " << iter << " nums " << w_X.cols() << ", num_gmm: " << num_gmm << std::endl;
    // Initial Guess
    std::vector<Eigen::VectorXd> init_mu;
    std::vector<Eigen::MatrixXd> init_sigma;
    for (int i = 0; i < num_gmm; i++) {
      int r = rand() % w_X.cols();

      Eigen::VectorXd random_pt_on_obj = w_X.col(r).topRows(3);
      init_mu.push_back(random_pt_on_obj);

      init_sigma.push_back(Eigen::Matrix3d::Identity() * 0.001);
    }

    // exec
    Eigen::MatrixXd res = w_X.topRows(3);
    // Initialize weights to uniform distribution
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(num_gmm, 1.0 / num_gmm);
    Eigen::MatrixXd prob = Eigen::MatrixXd::Zero(res.cols(), num_gmm);

    bool re = GMMFit::fit_multivariate(res, num_gmm, init_mu, init_sigma, weights, prob);

    if (!re) {
      continue;
    }

    std::cout << " init_mu : " << init_sigma[0] << std::endl;

    GMMPointCloud cell_gaussians;
    for (int i = 0; i < num_gmm; i++) {

      GMMPoint tmp_point;
      tmp_point.x = static_cast<float>(init_mu[i].x());
      tmp_point.y = static_cast<float>(init_mu[i].y());
      tmp_point.z = static_cast<float>(init_mu[i].z());
      cell_gaussians._points.push_back(tmp_point);

      GMMColor tmp_color;
      tmp_color.r = 34;
      tmp_color.g = 14;
      tmp_color.b = 221;
      cell_gaussians._colors.push_back(tmp_color);

      Eigen::Vector3d tmp_cov = init_sigma[i].diagonal();
      GMMScale covs;
      covs.cov_x = static_cast<float>(tmp_cov.x());
      covs.cov_y = static_cast<float>(tmp_cov.y());
      covs.cov_z = static_cast<float>(tmp_cov.z());
      cell_gaussians._scales.push_back(covs);
    }

    if (cell_gaussians._points.size() > 0) {
      Visualize3DGMMOpen3D(w_X, cell_gaussians);
    } else {
      std::cout << "No points in the cell" << std::endl;
    }
    iter++;
  }

  std::cout << "Done!" << std::endl;
  return 0;
}