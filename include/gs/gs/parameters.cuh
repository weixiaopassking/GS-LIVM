// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <filesystem>

namespace gs {
namespace param {

class OptimizationParameters {
 public:
  int empty_iterations;
  float position_lr_init;

  float scale_factor;
  float position_lr_final;
  float position_lr_delay_mult;
  int position_lr_max_steps;
  float feature_lr;
  float percent_dense;
  float opacity_lr;
  float scaling_lr;
  float rotation_lr;
  float lambda_dssim;
  float lambda_depth_simi;
  float lambda_delta_depth_simi;
  float min_opacity;
  int densification_interval;
  int opacity_reset_interval;
  int densify_from_iter;
  int densify_until_iter;
  float densify_grad_threshold;
  bool early_stopping;
  float convergence_threshold;
  bool empty_gpu_cache;
};

struct ModelParameters {
  int sh_degree = 0;
  std::filesystem::path output_path = "/home/xieys/projects/gslivo_ws/output";
  std::string images = "images";
  int resolution = -1;
  bool white_background = true;
  bool eval = false;
};

void Write_model_parameters_to_file(const gs::param::ModelParameters& params, std::string path);
}  // namespace param
}  // namespace gs
