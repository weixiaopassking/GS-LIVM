// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "gs/debug_utils.cuh"
#include "gs/rasterize_points.cuh"

struct GaussianRasterizationSettings {
  int image_height;
  int image_width;
  float tanfovx;
  float tanfovy;
  torch::Tensor bg;
  float scale_modifier;
  torch::Tensor viewmatrix;
  torch::Tensor projmatrix;
  int sh_degree;
  torch::Tensor camera_center;
  bool prefiltered;
};

class _RasterizeGaussians : public torch::autograd::Function<_RasterizeGaussians> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor means3D,
      torch::Tensor means2D,
      torch::Tensor sh,
      torch::Tensor colors_precomp,
      torch::Tensor opacities,
      torch::Tensor scales,
      torch::Tensor rotations,
      torch::Tensor cov3Ds_precomp,
      torch::Tensor image_height,
      torch::Tensor image_width,
      torch::Tensor tanfovx,
      torch::Tensor tanfovy,
      torch::Tensor bg,
      torch::Tensor scale_modifier,
      torch::Tensor viewmatrix,
      torch::Tensor projmatrix,
      torch::Tensor sh_degree,
      torch::Tensor camera_center,
      torch::Tensor prefiltered);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

class GaussianRasterizer : torch::nn::Module {
 public:
  GaussianRasterizer(GaussianRasterizationSettings raster_settings) : raster_settings_(raster_settings) {}

  torch::Tensor mark_visible(torch::Tensor positions);

  torch::autograd::tensor_list rasterize_gaussians(
      torch::Tensor means3D,
      torch::Tensor means2D,
      torch::Tensor sh,
      torch::Tensor colors_precomp,
      torch::Tensor opacities,
      torch::Tensor scales,
      torch::Tensor rotations,
      torch::Tensor cov3Ds_precomp,
      GaussianRasterizationSettings raster_settings);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor means3D,
      torch::Tensor means2D,
      torch::Tensor opacities,
      torch::Tensor shs = torch::Tensor(),
      torch::Tensor colors_precomp = torch::Tensor(),
      torch::Tensor scales = torch::Tensor(),
      torch::Tensor rotations = torch::Tensor(),
      torch::Tensor cov3D_precomp = torch::Tensor());

 private:
  GaussianRasterizationSettings raster_settings_;
};
