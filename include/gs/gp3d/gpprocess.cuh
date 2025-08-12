/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_GP3D_H_INCLUDED
#define CUDA_GP3D_H_INCLUDED

#include <torch/torch.h>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_CUDA
#include <gp3d/gp_types.h>
#include <gp3d/gpmap.h>
#include <liw/utility.h>

#include <glm/glm.hpp>

struct DColor {
  float r, g, b;
};

struct DPoint {
  float x, y, z, variance;
  int direction;
};

struct DPointGMM {
  float x, y, z, r, g, b, variance, gray;
  bool is_vaild;
};

struct DPointGMMRe {
  float x, y, z, r, g, b;
  std::vector<float> cov_;
};

struct DRegion {
  float x_min, x_max, y_min, y_max, z_min, z_max;
};

struct DVoxel {
  std::size_t hash_posi;
  int direction;
  int num_train;
  DRegion region;
  float mean;
  float interval;

  // points
  DPoint* points;

  // solve train
  float* iden_mat_train;

  float* global_train_points_variance;
  float* global_train_x_points;
  float* global_train_y_points;
  float* global_train_f_points;
  float* global_test_x;
  float* global_test_y;
  float* global_points_testm;

  float* ky;
  float* k_starm;
  float* kky;
  float* f_starm;
  float* k_variance;
};

class gpProcess : torch::nn::Module {
 public:
  gpProcess(GSLIVM::GpParameter _params) : gp_options_{_params} {
    num_gp_side = gp_options_.num_gp_side * gp_options_.neighbour_size;
    num_gp_side_square = num_gp_side * num_gp_side;
    std::cout << "<=== num_gp_side for mapping: " << gp_options_.num_gp_side << std::endl;
    std::cout << "<=== num_gp_side for init (x3): " << num_gp_side << std::endl;
    std::cout << "<=== num_gp_side_square (x3): " << num_gp_side_square << std::endl;
  }

  void forward_gp3d(
      std::vector<GSLIVM::needGPUdata>& data,
      std::vector<GSLIVM::varianceUpdate>& updateVas,
      GSLIVM::ImageRt& frame,
      std::vector<GSLIVM::GsForMap>& final_gs_sample,
      std::vector<GSLIVM::GsForLoss>& final_gs_calc_loss);

  void backward(torch::Tensor init);

  bool
  projectPointsToImage(const std::vector<Eigen::Vector3d>& points_incam, std::vector<DColor>& colors, cv::Mat& frame);
  Eigen::Vector3d transformRawPointToCamera(const Eigen::Vector3d& raw_point, GSLIVM::ImageRt& frame);
  bool getColors(const std::vector<Eigen::Vector3f>& points, std::vector<DColor>& colors, GSLIVM::ImageRt& frame);

  void copyDataArrayToGPU(DVoxel* h_voxels, DVoxel*& d_voxels, int num_voxels, int num_gp_side_square);

  void updateCamParams(float fx_, float fy_, float cx_, float cy_) {
    fx = fx_;
    fy = fy_;
    cx = cx_;
    cy = cy_;
  }

 public:
  int num_gp_side_square;
  int num_gp_side;

 private:
  GSLIVM::GpParameter gp_options_;
  float fx, fy, cx, cy;
  float d0 = 0;
  float d1 = 0;
  float d2 = 0;
  float d3 = 0;
  std::mutex gp_update_mutex;
  std::mutex final_gs_map_mutex;
  std::mutex final_gs_loss_mutex;

  std::vector<DVoxel> all_d_data;
  std::mutex mutex_added_final_gs_sample_insert;
  std::unordered_set<std::size_t> added_final_gs_sample;
};

#endif
