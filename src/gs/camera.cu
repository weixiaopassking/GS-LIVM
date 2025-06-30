#include <torch/torch.h>
#include <string>
#include <utility>
#include "gs/camera.cuh"

Camera::Camera(
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
    float scale)
    : _colmap_id(imported_colmap_id),
      _R(R),
      _T(T),
      _FoVx(FoVx),
      _FoVy(FoVy),
      _image_name(std::move(std::move(image_name))),
      _uid(uid),
      _scale(scale) {

  this->_original_image = torch::clamp(image, 0.f, 1.f);
  this->_image_width = this->_original_image.size(2);
  this->_image_height = this->_original_image.size(1);

  this->_zfar = 100.f;
  this->_znear = 0.01f;

  Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
  Tcw.block<3, 3>(0, 0) = _R.transpose();
  Tcw.block<3, 1>(0, 3) = -_R.transpose() * _T;
  torch::Tensor world_view_transform = torch::from_blob(Tcw.data(), {4, 4}, torch::kFloat32);
  this->_world_view_transform = world_view_transform.clone().to(torch::kCUDA, true);

  this->_projection_matrix =
      getProjectionMatrix(this->_znear, this->_zfar, this->_FoVx, this->_FoVy).to(torch::kCUDA, true);

  this->_full_proj_transform =
      this->_world_view_transform.unsqueeze(0).bmm(this->_projection_matrix.unsqueeze(0)).squeeze(0);

  this->_camera_center = this->_world_view_transform.inverse()[3].slice(0, 0, 3);

  this->_K = Eigen::Matrix3f::Identity();
  this->_K(0, 0) = f_x;
  this->_K(1, 1) = f_y;
  this->_K(0, 2) = c_x;
  this->_K(1, 2) = c_y;
}

torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY) {
  float tanHalfFovY = std::tan((fovY / 2.f));
  float tanHalfFovX = std::tan((fovX / 2.f));

  float top = tanHalfFovY * znear;
  float bottom = -top;
  float right = tanHalfFovX * znear;
  float left = -right;

  Eigen::Matrix4f P = Eigen::Matrix4f::Zero();

  float z_sign = 1.f;

  P(0, 0) = 2.f * znear / (right - left);
  P(1, 1) = 2.f * znear / (top - bottom);
  P(0, 2) = (right + left) / (right - left);
  P(1, 2) = (top + bottom) / (top - bottom);
  P(3, 2) = z_sign;
  P(2, 2) = z_sign * zfar / (zfar - znear);
  P(2, 3) = -(zfar * znear) / (zfar - znear);

  // create torch::Tensor from Eigen::Matrix
  auto PTensor = torch::from_blob(P.data(), {4, 4}, torch::kFloat32);
  // clone the tensor to allocate new memory
  return PTensor.clone();
}

float fov2focal(float fov, int pixels) {
  return static_cast<float>(pixels) / (2.f * std::tan(fov / 2.f));
}

float focal2fov(float focal, int pixels) {
  return 2 * std::atan(static_cast<float>(pixels) / (2.f * focal));
}

// Eigen::Matrix3f qvec2rotmat(const Eigen::Quaternionf& q) {
//   Eigen::Vector4f qvec = q.coeffs();  // [x, y, z, w]

//   Eigen::Matrix3f rotmat;
//   rotmat << 1.f - 2.f * qvec[2] * qvec[2] - 2.f * qvec[3] * qvec[3], 2.f * qvec[1] * qvec[2] - 2.f * qvec[0] * qvec[3],
//       2.f * qvec[3] * qvec[1] + 2.f * qvec[0] * qvec[2], 2.f * qvec[1] * qvec[2] + 2.f * qvec[0] * qvec[3],
//       1.f - 2.f * qvec[1] * qvec[1] - 2.f * qvec[3] * qvec[3], 2.f * qvec[2] * qvec[3] - 2.f * qvec[0] * qvec[1],
//       2.f * qvec[3] * qvec[1] - 2.f * qvec[0] * qvec[2], 2.f * qvec[2] * qvec[3] + 2.f * qvec[0] * qvec[1],
//       1.f - 2.f * qvec[1] * qvec[1] - 2.f * qvec[2] * qvec[2];

//   return rotmat;
// }

// Eigen::Quaternionf rotmat2qvec(const Eigen::Matrix3f& R) {
//   Eigen::Quaternionf qvec(R);
//   // the order of coefficients is different in python implementation.
//   // Might be a bug here if data comes in wrong order! TODO: check
//   if (qvec.w() < 0.f) {
//     qvec.coeffs() *= -1.f;
//   }
//   return qvec;
// }

// torch::Tensor getWorld2View2(
//     const Eigen::Matrix3f& R,
//     const Eigen::Vector3f& t,
//     const Eigen::Vector3f& translate /*= Eigen::Vector3d::Zero()*/,
//     float scale /*= 1.0*/) {
//   Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
//   Rt.block<3, 3>(0, 0) = R.transpose();
//   Rt.block<3, 1>(0, 3) = -R.transpose() * t;

//   Eigen::Matrix4f C2W = Rt.inverse();
//   Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
//   cam_center = (cam_center + translate) * scale;
//   C2W.block<3, 1>(0, 3) = cam_center;
//   Rt = C2W.inverse();
//   // Here we create a torch::Tensor from the Eigen::Matrix
//   // Note that the tensor will be on the CPU, you may want to move it to the desired device later
//   torch::Tensor RtTensor = torch::from_blob(Rt.data(), {4, 4}, torch::kFloat32);
//   // clone the tensor to allocate new memory, as from_blob shares the same memory
//   // this step is important if Rt will go out of scope and the tensor will be used later
//   return RtTensor.clone();
// }

// Eigen::Matrix4f getWorld2View2Eigen(
//     const Eigen::Matrix3f& R,
//     const Eigen::Vector3f& t,
//     const Eigen::Vector3f& translate /*= Eigen::Vector3d::Zero()*/,
//     float scale /*= 1.0*/) {
//   Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
//   Rt.block<3, 3>(0, 0) = R.transpose();
//   Rt.block<3, 1>(0, 3) = -R.transpose() * t;
//   // Rt.block<3, 1>(0, 3) = t;

//   Eigen::Matrix4f C2W = Rt.inverse();
//   Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
//   cam_center = (cam_center + translate) * scale;
//   C2W.block<3, 1>(0, 3) = cam_center;
//   Rt = C2W.inverse();
//   return Rt;
// }