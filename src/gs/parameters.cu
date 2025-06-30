// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "gs/parameters.cuh"

namespace gs {
namespace param {

void Write_model_parameters_to_file(const gs::param::ModelParameters& params, std::string path) {
  std::filesystem::path outputPath(path);
  // Make sure the directory exists
  std::filesystem::create_directories(outputPath);

  std::ofstream cfg_log_f(outputPath / "cfg_args");
  if (!cfg_log_f.is_open()) {
    std::cerr << "Failed to open file for writing!" << std::endl;
    return;
  }

  // Write the parameters in the desired format
  cfg_log_f << "Namespace(";
  cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
  cfg_log_f << "images='" << params.images << "', ";
  cfg_log_f << "model_path='" << params.output_path.string() << "', ";
  cfg_log_f << "resolution=" << params.resolution << ", ";
  cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
  cfg_log_f << "source_path='" << params.output_path.string() << "', ";
  cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
  cfg_log_f.close();

  // std::cout << "Output folder: " << params.output_path.string() << std::endl;
}

}  // namespace param
}  // namespace gs
