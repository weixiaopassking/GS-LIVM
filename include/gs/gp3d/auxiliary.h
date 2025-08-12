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

#ifndef CUDA_GP3D_AUXILIARY_H_INCLUDED
#define CUDA_GP3D_AUXILIARY_H_INCLUDED

#include "stdio.h"

#define CHECK_CUDA(A, debug)                                                                                     \
  A;                                                                                                             \
  if (debug) {                                                                                                   \
    auto ret = cudaDeviceSynchronize();                                                                          \
    if (ret != cudaSuccess) {                                                                                    \
      std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
      throw std::runtime_error(cudaGetErrorString(ret));                                                         \
    }                                                                                                            \
  }

#endif