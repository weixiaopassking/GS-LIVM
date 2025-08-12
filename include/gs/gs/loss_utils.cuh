// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <torch/torch.h>
#include <cmath>

namespace gaussian_splatting {
static const float C1 = 0.01 * 0.01;
static const float C2 = 0.03 * 0.03;

torch::Tensor l1_loss(const torch::Tensor& network_output, const torch::Tensor& gt) {
  return torch::abs((network_output - gt)).mean();
}

torch::Tensor inv_depth(const torch::Tensor& depth, float epsilon = 1e-2) {
  auto mask = depth <= epsilon;
  auto depth_clamped = depth.clamp(epsilon);
  auto inverse_depth = 1.0 / depth_clamped;
  inverse_depth.masked_fill_(mask, 0);
  return inverse_depth;
}

// 1D Gaussian kernel
torch::Tensor gaussian(int window_size, float sigma) {
  torch::Tensor gauss = torch::empty(window_size);
  for (int x = 0; x < window_size; ++x) {
    gauss[x] = std::exp(-(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) / (2.f * sigma * sigma));
  }
  return gauss / gauss.sum();
}

torch::Tensor create_window(int window_size, int channel) {
  auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
  auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
  return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
}

// Image Quality Assessment: From Error Visibility to
// Structural Similarity (SSIM), Wang et al. 2004
// The SSIM value lies between -1 and 1, where 1 means perfect similarity.
// It's considered a better metric than mean squared error for perceptual image quality as it considers changes in structural information,
// luminance, and contrast.
torch::Tensor
ssim(const torch::Tensor& img1, const torch::Tensor& img2, const torch::Tensor& window, int window_size, int channel) {
  auto mu1 = torch::nn::functional::conv2d(
      img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
  auto mu1_sq = mu1.pow(2);
  auto sigma1_sq =
      torch::nn::functional::conv2d(
          img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) -
      mu1_sq;

  auto mu2 = torch::nn::functional::conv2d(
      img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
  auto mu2_sq = mu2.pow(2);
  auto sigma2_sq =
      torch::nn::functional::conv2d(
          img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) -
      mu2_sq;

  auto mu1_mu2 = mu1 * mu2;
  auto sigma12 =
      torch::nn::functional::conv2d(
          img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) -
      mu1_mu2;
  auto ssim_map =
      ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

  return ssim_map.mean();
}

// 使用高斯滤波进行平滑
torch::Tensor smoothDepth(torch::Tensor& depthTensor) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto gaussian_kernel =
      torch::tensor({{1.0, 2.0, 1.0}, {2.0, 4.0, 2.0}, {1.0, 2.0, 1.0}}, options).to(torch::kCUDA, true) / 16.0;

  gaussian_kernel.set_requires_grad(false);

  auto gaussian_kernel_reshaped = gaussian_kernel.reshape({1, 1, 3, 3});

  auto depth_tensor_reshaped = depthTensor.reshape({1, 1, depthTensor.size(0), depthTensor.size(1)});

  auto smoothed_depth_tensor = torch::conv2d(depth_tensor_reshaped, gaussian_kernel_reshaped, {}, 1, 1);

  return torch::abs((smoothed_depth_tensor.squeeze() - depthTensor)).mean();
}

torch::Tensor psnr(torch::Tensor& rendered_img, torch::Tensor& gt_img) {
  torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
  torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
  return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean();
}

}  // namespace gaussian_splatting
