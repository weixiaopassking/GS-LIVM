#pragma once

#pragma diag_suppress code_of_warning
#include <Eigen/Dense>
#pragma diag_default code_of_warning
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

#include "gs/parameters.cuh"
#include "gs/stb_image.h"
#include "gs/stb_image_resize.h"

// enum class CAMERA_MODEL {
//   SIMPLE_PINHOLE = 0,
//   PINHOLE = 1,
//   SIMPLE_RADIAL = 2,
//   RADIAL = 3,
//   OPENCV = 4,
//   OPENCV_FISHEYE = 5,
//   FULL_OPENCV = 6,
//   FOV = 7,
//   SIMPLE_RADIAL_FISHEYE = 8,
//   RADIAL_FISHEYE = 9,
//   THIN_PRISM_FISHEYE = 10,
//   UNDEFINED = 11
// };

// struct CameraInfo {
//   uint32_t _camera_ID;
//   Eigen::Matrix3f _R;  // rotation  matrix
//   Eigen::Vector3f _T;  // translation vector
//   float _fov_x;
//   float _fov_y;
//   std::string _image_name;
//   std::filesystem::path _image_path;
//   CAMERA_MODEL _camera_model;
//   int _width;
//   int _height;
//   int _img_w;
//   int _img_h;
//   int _channels;
//   std::vector<double> _params;
// };

class Camera : torch::nn::Module {
 public:
  Camera(
      int imported_colmap_id,
      Eigen::Matrix3f R,
      Eigen::Vector3f T,
      float f_x,
      float f_y,
      float c_x,
      float c_y,
      float FoVx,
      float FoVy,
      torch::Tensor image,
      std::string image_name,
      int uid,
      float scale = 1.f);

  // Getters
  int Get_colmap_id() const { return _colmap_id; }

  Eigen::Matrix3f& Get_R() { return _R; }

  Eigen::Vector3f& Get_T() { return _T; }

  float Get_FoVx() const { return static_cast<float>(_FoVx); }

  float Get_FoVy() const { return static_cast<float>(_FoVy); }

  std::string Get_image_name() const { return _image_name; }

  const torch::Tensor& Get_original_image() { return _original_image; }

  int Get_image_width() const { return _image_width; }

  int Get_image_height() const { return _image_height; }

  float Get_zfar() const { return _zfar; }

  float Get_znear() const { return _znear; }

  torch::Tensor& Get_world_view_transform() { return _world_view_transform; }

  torch::Tensor& Get_projection_matrix() { return _projection_matrix; }

  Eigen::Matrix3f& Get_K() { return _K; }

  torch::Tensor& Get_full_proj_transform() { return _full_proj_transform; }

  torch::Tensor& Get_camera_center() { return _camera_center; }

 private:
  int _uid;
  int _colmap_id;
  Eigen::Matrix3f _R;  // rotation  matrix
  Eigen::Vector3f _T;  // translation vector
  float _FoVx;
  float _FoVy;
  std::string _image_name;
  torch::Tensor _original_image;
  int _image_width;
  int _image_height;
  float _zfar;
  float _znear;
  torch::Tensor _trans;
  float _scale;
  torch::Tensor _world_view_transform;
  torch::Tensor _projection_matrix;
  torch::Tensor _full_proj_transform;
  torch::Tensor _camera_center;

  Eigen::Matrix3f _K;
};

namespace gs::param {
struct ModelParameters;
}

// torch::Tensor getWorld2View2(
//     const Eigen::Matrix3f& R,
//     const Eigen::Vector3f& t,
//     const Eigen::Vector3f& translate = Eigen::Vector3f::Zero(),
//     float scale = 1.0);

// TODO: hacky. Find better way
// Eigen::Matrix4f getWorld2View2Eigen(
//     const Eigen::Matrix3f& R,
//     const Eigen::Vector3f& t,
//     const Eigen::Vector3f& translate = Eigen::Vector3f::Zero(),
//     float scale = 1.0);

torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY);

float fov2focal(float fov, int pixels);

float focal2fov(float focal, int pixels);

// Eigen::Matrix3f qvec2rotmat(const Eigen::Quaternionf& qvec);

// Eigen::Quaternionf rotmat2qvec(const Eigen::Matrix3f& R);
