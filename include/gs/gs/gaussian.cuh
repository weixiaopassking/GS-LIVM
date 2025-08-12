// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include "camera.cuh"
#include "general_utils.cuh"
#include "parameters.cuh"
#include "sh_utils.cuh"

#include <pcl/io/ply_io.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tinyply.h>

#include <gp3d/gp_types.h>

class GaussianModel {
 public:
  explicit GaussianModel(int sh_degree);
  // Copy constructor
  GaussianModel(const GaussianModel& other) = delete;
  // Copy assignment operator
  GaussianModel& operator=(const GaussianModel& other) = delete;
  // Move constructor
  GaussianModel(GaussianModel&& other) = default;
  // Move assignment operator
  GaussianModel& operator=(GaussianModel&& other) = default;

 public:
  inline void Incre_gs_cache_size(int _para) { _gs_cache_size += _para; }

  inline int Get_gs_cache_size() const { return _gs_cache_size; }

  // Getters
  inline torch::Tensor Get_xyz() const { return _xyz; }

  inline torch::Tensor Get_opacity() const { return torch::sigmoid(_opacity); }

  inline torch::Tensor Get_rotation() const { return torch::nn::functional::normalize(_rotation); }

  inline torch::Tensor Get_features() const {
    auto features_dc = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
  }

  int Get_max_sh_degree() const { return _max_sh_degree; }

  torch::Tensor Get_scaling() { return torch::exp(_scaling); }

  void Create_from_pcd(const gs::param::OptimizationParameters& params, GSLIVM::GsForMaps& pcd, float spatial_lr_scale);

  void addNewPointcloud(
      const gs::param::OptimizationParameters& params,
      GSLIVM::GsForMaps& pcd,
      int& iter,
      float spatial_lr_scale);

  void Training_setup(const gs::param::OptimizationParameters& params);

  void Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration);
  //   std::pair<Eigen::Vector3f, float> getNerfppNorm(std::vector<CameraInfo>& cam_info);

  bool calcSimiLoss(GSLIVM::GsForLosses& items, torch::Tensor& loss, float& lambda);
  torch::Tensor calcDeltaSimi(GSLIVM::DeltaSimi& cam, GSLIVM::DeltaSimi& cam_ref);

 public:
  std::unique_ptr<torch::optim::Adam> _optimizer;
  torch::Tensor _max_radii2D;

 private:
  void densification_postfix(
      torch::Tensor& new_xyz,
      torch::Tensor& new_features_dc,
      torch::Tensor& new_features_rest,
      torch::Tensor& new_scaling,
      torch::Tensor& new_rotation,
      torch::Tensor& new_opacity);

  std::vector<std::string> construct_list_of_attributes();

  void prune_optimizer(
      torch::optim::Adam* optimizer,
      const torch::Tensor& mask,
      torch::Tensor& old_tensor,
      int param_position);

  void cat_tensors_to_optimizer(
      torch::optim::Adam* optimizer,
      torch::Tensor& extension_tensor,
      torch::Tensor& old_tensor,
      int param_position);

  torch::Tensor rotation_matrix_to_quaternion(const torch::Tensor& rotation_matrix);
  void decomposeSR(torch::Tensor& covs, torch::Tensor& scale_p, torch::Tensor& quat_p);
  torch::Tensor compute_min_distance(
      const torch::Tensor& points,
      const torch::Tensor& spheres_positions,
      const torch::Tensor& scales);

 private:
  int _active_sh_degree = 0;
  int _max_sh_degree = 0;
  float _spatial_lr_scale = 0.f;
  float _percent_dense = 0.f;
  float _max_clip_scaling = 0.005f;

  int _gs_cache_size = 0;

  torch::Tensor _denom;
  torch::Tensor _xyz;
  torch::Tensor _features_dc;
  torch::Tensor _features_rest;
  torch::Tensor _scaling;
  torch::Tensor _rotation;
  torch::Tensor _xyz_gradient_accum;
  torch::Tensor _opacity;

  std::unordered_map<std::size_t, std::vector<int>> gs_hash_indexes_;
};

void Write_output_ply(
    const std::filesystem::path& file_path,
    const std::vector<torch::Tensor>& tensors,
    const std::vector<std::string>& attributes_names);