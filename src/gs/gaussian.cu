#include <exception>
#include <ranges>
#include <thread>

#include "gs/debug_utils.cuh"
#include "gs/gaussian.cuh"

GaussianModel::GaussianModel(int sh_degree) : _max_sh_degree(sh_degree) {}

void GaussianModel::decomposeSR(torch::Tensor& covs, torch::Tensor& scale_p, torch::Tensor& quat_p) {
  scale_p = covs.diagonal(0, -2, -1);

  // seems to be useless
  // auto svd_result = torch::linalg_svd(covs);

  // auto rot_p = std::get<0>(svd_result);

  // TORCH_CHECK(rot_p.dim() == 3, "Input rotation_matrix should be of shape (n, 3, 3)");
  // TORCH_CHECK(rot_p.size(1) == 3 && rot_p.size(2) == 3, "Each rotation matrix should be of shape (3, 3)");
  // int n = rot_p.size(0);
  // quat_p = torch::empty({n, 4}, rot_p.options());

  // auto trace = rot_p.select(1, 0).select(1, 0) + rot_p.select(1, 1).select(1, 1) + rot_p.select(1, 2).select(1, 2);

  // auto mask1 = trace > 0.0f;
  // auto s1 = torch::sqrt(trace + 1.0f) * 2.0f;  // s = 4 * qw
  // auto qw1 = 0.25f * s1;
  // auto qx1 = (rot_p.select(1, 2).select(1, 1) - rot_p.select(1, 1).select(1, 2)) / s1;
  // auto qy1 = (rot_p.select(1, 0).select(1, 2) - rot_p.select(1, 2).select(1, 0)) / s1;
  // auto qz1 = (rot_p.select(1, 1).select(1, 0) - rot_p.select(1, 0).select(1, 1)) / s1;

  // auto mask2 = (rot_p.select(1, 0).select(1, 0) > rot_p.select(1, 1).select(1, 1)) &
  //              (rot_p.select(1, 0).select(1, 0) > rot_p.select(1, 2).select(1, 2));
  // auto s2 =
  //     torch::sqrt(
  //         1.0f + rot_p.select(1, 0).select(1, 0) - rot_p.select(1, 1).select(1, 1) - rot_p.select(1, 2).select(1, 2)) *
  //     2.0f;  // s = 4 * qx
  // auto qw2 = (rot_p.select(1, 2).select(1, 1) - rot_p.select(1, 1).select(1, 2)) / s2;
  // auto qx2 = 0.25f * s2;
  // auto qy2 = (rot_p.select(1, 0).select(1, 1) + rot_p.select(1, 1).select(1, 0)) / s2;
  // auto qz2 = (rot_p.select(1, 0).select(1, 2) + rot_p.select(1, 2).select(1, 0)) / s2;

  // auto mask3 = (rot_p.select(1, 1).select(1, 1) > rot_p.select(1, 2).select(1, 2));
  // auto s3 =
  //     torch::sqrt(
  //         1.0f + rot_p.select(1, 1).select(1, 1) - rot_p.select(1, 0).select(1, 0) - rot_p.select(1, 2).select(1, 2)) *
  //     2.0f;  // s = 4 * qy
  // auto qw3 = (rot_p.select(1, 0).select(1, 2) - rot_p.select(1, 2).select(1, 0)) / s3;
  // auto qx3 = (rot_p.select(1, 0).select(1, 1) + rot_p.select(1, 1).select(1, 0)) / s3;
  // auto qy3 = 0.25f * s3;
  // auto qz3 = (rot_p.select(1, 1).select(1, 2) + rot_p.select(1, 2).select(1, 1)) / s3;

  // auto s4 =
  //     torch::sqrt(
  //         1.0f + rot_p.select(1, 2).select(1, 2) - rot_p.select(1, 0).select(1, 0) - rot_p.select(1, 1).select(1, 1)) *
  //     2.0f;  // s = 4 * qz
  // auto qw4 = (rot_p.select(1, 1).select(1, 0) - rot_p.select(1, 0).select(1, 1)) / s4;
  // auto qx4 = (rot_p.select(1, 0).select(1, 2) + rot_p.select(1, 2).select(1, 0)) / s4;
  // auto qy4 = (rot_p.select(1, 1).select(1, 2) + rot_p.select(1, 2).select(1, 1)) / s4;
  // auto qz4 = 0.25f * s4;

  // quat_p.index_put_({mask1, 0}, qx1.masked_select(mask1));
  // quat_p.index_put_({mask1, 1}, qy1.masked_select(mask1));
  // quat_p.index_put_({mask1, 2}, qz1.masked_select(mask1));
  // quat_p.index_put_({mask1, 3}, qw1.masked_select(mask1));

  // quat_p.index_put_({mask2, 0}, qx2.masked_select(mask2));
  // quat_p.index_put_({mask2, 1}, qy2.masked_select(mask2));
  // quat_p.index_put_({mask2, 2}, qz2.masked_select(mask2));
  // quat_p.index_put_({mask2, 3}, qw2.masked_select(mask2));

  // quat_p.index_put_({mask3, 0}, qx3.masked_select(mask3));
  // quat_p.index_put_({mask3, 1}, qy3.masked_select(mask3));
  // quat_p.index_put_({mask3, 2}, qz3.masked_select(mask3));
  // quat_p.index_put_({mask3, 3}, qw3.masked_select(mask3));

  // quat_p.index_put_(
  //     {torch::logical_not(mask1 | mask2 | mask3), 0}, qx4.masked_select(torch::logical_not(mask1 | mask2 | mask3)));
  // quat_p.index_put_(
  //     {torch::logical_not(mask1 | mask2 | mask3), 1}, qy4.masked_select(torch::logical_not(mask1 | mask2 | mask3)));
  // quat_p.index_put_(
  //     {torch::logical_not(mask1 | mask2 | mask3), 2}, qz4.masked_select(torch::logical_not(mask1 | mask2 | mask3)));
  // quat_p.index_put_(
  //     {torch::logical_not(mask1 | mask2 | mask3), 3}, qw4.masked_select(torch::logical_not(mask1 | mask2 | mask3)));
}

torch::Tensor GaussianModel::compute_min_distance(
    const torch::Tensor& points,
    const torch::Tensor& spheres_positions,
    const torch::Tensor& scales) {

  assert(spheres_positions.requires_grad());
  assert(scales.requires_grad());

  auto radius = scales.mean();

  auto m = points.size(0);
  auto n = spheres_positions.size(0);

  auto points_expanded = points.unsqueeze(1).expand({m, n, 3});
  auto spheres_positions_expanded = spheres_positions.unsqueeze(0).expand({m, n, 3});

  auto relative_positions = points_expanded - spheres_positions_expanded;

  auto distances_to_centers = relative_positions.norm(2, 2);

  auto distances_to_surface = distances_to_centers - radius;

  auto distances_to_surface_clamped = torch::max(distances_to_surface, torch::zeros_like(distances_to_surface));

  auto [min_distances, _] = distances_to_surface_clamped.min(1);

  return min_distances.mean();
}

torch::Tensor GaussianModel::calcDeltaSimi(GSLIVM::DeltaSimi& cam, GSLIVM::DeltaSimi& cam_ref) {
  int height = cam.depth.size(1);
  int width = cam.depth.size(2);

  // 创建网格坐标
  auto meshgrid_x = torch::arange(0, width, torch::kFloat32).to(torch::kCUDA, true).repeat({height, 1});
  auto meshgrid_y = torch::arange(0, height, torch::kFloat32).to(torch::kCUDA, true).unsqueeze(1).repeat({1, width});

  // 创建像素坐标
  auto pix_coords = torch::stack({meshgrid_x.flatten(), meshgrid_y.flatten()}, 0);
  auto ones = torch::ones({1, pix_coords.size(1)}, torch::kFloat32).to(torch::kCUDA, true);
  auto hom_pix_coords = torch::cat({pix_coords, ones}, 0).to(torch::kCUDA, true);

  auto inv_K =
      torch::from_blob(const_cast<float*>(cam.inv_K.data()), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(torch::kCUDA, true);

  auto d_uv = hom_pix_coords * cam.depth.flatten();

  auto cam_points = torch::matmul(inv_K, d_uv).to(torch::kCUDA, true);
  cam_points = torch::cat({cam_points, ones}, 0);

  auto cam_pose_R_tensor =
      torch::from_blob(const_cast<float*>(cam.cam_pose_R.data()), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(torch::kCUDA, true);
  auto cam_pose_t_tensor =
      torch::from_blob(const_cast<float*>(cam.cam_pose_t.data()), {3}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(torch::kCUDA, true);

  auto cam_ref_pose_R_tensor =
      torch::from_blob(
          const_cast<float*>(cam_ref.cam_pose_R.data()), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(torch::kCUDA, true);
  auto cam_ref_t_tensor =
      torch::from_blob(
          const_cast<float*>(cam_ref.cam_pose_t.data()), {3}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(torch::kCUDA, true);

  torch::Tensor T = torch::eye(4, torch::kFloat32).to(torch::kCUDA, true);
  T.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, cam_pose_R_tensor);
  T.index_put_({torch::indexing::Slice(0, 3), 3}, cam_pose_t_tensor);

  torch::Tensor T_ref = torch::eye(4, torch::kFloat32).to(torch::kCUDA, true);
  T_ref.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, cam_ref_pose_R_tensor);
  T_ref.index_put_({torch::indexing::Slice(0, 3), 3}, cam_ref_t_tensor);

  torch::Tensor T_trans = torch::matmul(T_ref, torch::inverse(T)).to(torch::kCUDA, true);
  auto proj_cam_points = torch::matmul(T_trans, cam_points);

  auto ref_K =
      torch::from_blob(const_cast<float*>(cam_ref.K.data()), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(torch::kCUDA, true);
  auto proj_2d = torch::matmul(ref_K, proj_cam_points.index({torch::indexing::Slice(0, 3)}));
  auto pix_coords_ref = proj_2d.index({torch::indexing::Slice(0, 2)}) / proj_2d.index({2}).unsqueeze(0);

  auto depth_values = proj_cam_points.index({2});

  pix_coords_ref = pix_coords_ref.view({2, height, width}).permute({1, 2, 0});

  // Normalizing pixel coordinates
  pix_coords_ref.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), 0},
      pix_coords_ref.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) / (width - 1) * 2 - 1);
  pix_coords_ref.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), 1},
      pix_coords_ref.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) / (height - 1) * 2 - 1);

  // Creating grid for sampling
  auto grid = pix_coords_ref.unsqueeze(0);  // Shape should be [1, height, width, 2]

  // Setting grid_sample options
  auto options = torch::nn::functional::GridSampleFuncOptions()
                     .mode(torch::kBilinear)
                     .padding_mode(torch::kZeros)
                     .align_corners(true);

  // Performing grid sample on depth values
  torch::Tensor output_depth = torch::nn::functional::grid_sample(
      depth_values.view({1, 1, height, width}).to(torch::kCUDA, true), grid.to(torch::kCUDA, true), options);

  // Convert back to 2D depth map
  auto output = output_depth.squeeze(0).squeeze(0);
  return output;
}

bool GaussianModel::calcSimiLoss(GSLIVM::GsForLosses& losses, torch::Tensor& loss, float& lambda) {
  std::vector<int> mask_indexes;
  torch::Tensor points = torch::empty({0, 3}, torch::kFloat32);

  for (const auto& [key, tensor] : losses._losses) {
    if (gs_hash_indexes_.count(key) > 0) {
      auto& curr_index = gs_hash_indexes_[key];
      mask_indexes.reserve(mask_indexes.size() + curr_index.size());
      mask_indexes.insert(mask_indexes.end(), curr_index.begin(), curr_index.end());
      points = torch::cat({points, tensor}, 0);
    }
  }

  if (points.size(0) == 0) {
    return false;
  }

  auto mask_tensor = torch::from_blob(mask_indexes.data(), {static_cast<long>(mask_indexes.size())}, torch::kInt)
                         .to(torch::kLong)
                         .to(torch::kCUDA, true);

  auto loss_mask = torch::zeros({this->Get_xyz().size(0)}, torch::kLong).to(torch::kCUDA);

  loss_mask.scatter_(0, mask_tensor, 1);
  auto xyz_for_loss = this->Get_xyz().index_select(0, loss_mask.nonzero().squeeze());
  auto scale_for_loss = this->Get_scaling().index_select(0, loss_mask.nonzero().squeeze());

  torch::Tensor selected_points;
  if (points.size(0) < MAX_SIMI) {
    selected_points = points.to(torch::kCUDA, true);
  } else {
    torch::Tensor random_indices = torch::randperm(points.size(0));
    torch::Tensor selected_indices_simi = random_indices.slice(0, 0, MAX_SIMI);
    selected_points = points.index_select(0, selected_indices_simi).to(torch::kCUDA, true);
  }

  loss = lambda * compute_min_distance(selected_points, xyz_for_loss, scale_for_loss);
  return true;
}

void GaussianModel::addNewPointcloud(
    const gs::param::OptimizationParameters& params,
    GSLIVM::GsForMaps& pcd,
    int& iter,
    float spatial_lr_scale) {
  _spatial_lr_scale = spatial_lr_scale;

  torch::Tensor new_xyz = pcd.gs_xyzs.slice(0, 0, pcd.gs_xyzs.size(0)).to(torch::kCUDA, true);

  // assert
  if (pcd.indexes.back().back() != _xyz.size(0) + new_xyz.size(0) - 1) {
    std::cout << "<<< ERROR: indexing mismatch in gaussian addNewPointcloud: " << pcd.indexes.back().back()
              << " != " << _xyz.size(0) << " + " << new_xyz.size(0) - 1 << std::endl;
    exit(-1);
  }

  for (int index = 0; index < pcd.hash_posi_s.size(); index++) {
    auto [it, interted] = gs_hash_indexes_.try_emplace(pcd.hash_posi_s[index], pcd.indexes[index]);
    if (!interted) {
      std::cout << "<<< ERROR:  keys duplicated in gaussian addNewPointcloud : " << pcd.hash_posi_s[index] << std::endl;
      exit(-1);
    }
  }

  // scale & rotation
  torch::Tensor new_scaling;
  {
    // // no initialization of gaussian, uncommont this to verify the code
    // torch::Tensor all_xyz = torch::cat({new_xyz, _xyz}, 0);
    // auto dist = torch::clamp_max(torch::clamp_min(distCUDA2(all_xyz), 0.0000001), _max_clip_scaling);
    // auto dist2 = dist.slice(0, 0, new_xyz.size(0));
    // new_scaling = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3}).to(torch::kCUDA, true);
  }
  {
    // initialization of gaussian of gs-livom version
    torch::Tensor scale_p;
    torch::Tensor rot_p;
    decomposeSR(pcd.gs_covs, scale_p, rot_p);

    new_scaling = torch::log(torch::sqrt(scale_p * params.scale_factor)).to(torch::kCUDA, true);
  }
  torch::Tensor new_rotation =
      torch::zeros({new_xyz.size(0), 4}).index_put_({torch::indexing::Slice(), 0}, 1).to(torch::kCUDA, true);
  // torch::Tensor new_rotation = rot_p.to(torch::kCUDA, true);

  // opacity
  torch::Tensor new_opacity = inverse_sigmoid(0.5 * torch::ones({new_xyz.size(0), 1})).to(torch::kCUDA, true);
  // colors
  auto fused_color = RGB2SH(pcd.gs_rgbs / 255.f);

  // features
  auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))})
                      .to(torch::kCUDA, true);
  features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
  features.index_put_(
      {torch::indexing::Slice(),
       torch::indexing::Slice(3, torch::indexing::None),
       torch::indexing::Slice(1, torch::indexing::None)},
      0.0);
  torch::Tensor new_features_dc =
      features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous()
          .to(torch::kCUDA, true);
  torch::Tensor new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)})
          .transpose(1, 2)
          .contiguous()
          .to(torch::kCUDA, true);

  densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);
}

/**
 * @brief Initialize Gaussian Model from a Point Cloud.
 *
 * This function creates a Gaussian model from a given PointCloud object. It also sets
 * the spatial learning rate scale. The model's features, scales, rotations, and opacities
 * are initialized based on the input point cloud.
 *
 * @param pcd The input point cloud
 * @param spatial_lr_scale The spatial learning rate scale
 */
void GaussianModel::Create_from_pcd(
    const gs::param::OptimizationParameters& params,
    GSLIVM::GsForMaps& pcd,
    float spatial_lr_scale) {
  _spatial_lr_scale = spatial_lr_scale;

  _xyz = pcd.gs_xyzs.to(torch::kCUDA).set_requires_grad(true);

  if (pcd.indexes.back().back() != _xyz.size(0) - 1) {
    std::cout << "<<< ERROR: indexing mismatch in gaussian Create_from_pcd: " << pcd.indexes.back().back()
              << " != " << _xyz.size(0) - 1 << std::endl;
    exit(-1);
  }

  for (int index = 0; index < pcd.hash_posi_s.size(); index++) {
    auto [it, interted] = gs_hash_indexes_.try_emplace(pcd.hash_posi_s[index], pcd.indexes[index]);
    if (!interted) {
      std::cout << "<<< ERROR:  keys duplicated in gaussian Create_from_pcd" << std::endl;
      exit(-1);
    }
  }

  // scale & rotation
  torch::Tensor scale_p;
  torch::Tensor rot_p;
  decomposeSR(pcd.gs_covs, scale_p, rot_p);

  // torch::Tensor scale_climp = torch::clamp_max(torch::clamp_min(scale_p, 0.000001), _max_clip_scaling);
  _scaling = torch::log(torch::sqrt(scale_p * params.scale_factor)).to(torch::kCUDA, true).set_requires_grad(true);
  _rotation = torch::zeros({_xyz.size(0), 4})
                  .index_put_({torch::indexing::Slice(), 0}, 1)
                  .to(torch::kCUDA, true)
                  .set_requires_grad(true);
  // _rotation = rot_p.to(torch::kCUDA, true).set_requires_grad(true);

  // opacities
  _opacity = inverse_sigmoid(0.5 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true).set_requires_grad(true);
  // _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

  // colors
  auto fused_color = RGB2SH(pcd.gs_rgbs / 255.f);

  // features
  auto features =
      torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
  features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
  features.index_put_(
      {torch::indexing::Slice(),
       torch::indexing::Slice(3, torch::indexing::None),
       torch::indexing::Slice(1, torch::indexing::None)},
      0.0);
  _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)})
                     .transpose(1, 2)
                     .contiguous()
                     .set_requires_grad(true);
  _features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)})
          .transpose(1, 2)
          .contiguous()
          .set_requires_grad(true);
}

/**
 * @brief Setup the Gaussian Model for training
 *
 * This function sets up the Gaussian model for training by initializing several
 * parameters and settings based on the provided OptimizationParameters object.
 *
 * @param params The OptimizationParameters object providing the settings for training
 */
void GaussianModel::Training_setup(const gs::param::OptimizationParameters& params) {
  this->_percent_dense = params.percent_dense;
  // this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
  // this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);

  std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups;
  optimizer_params_groups.reserve(6);
  optimizer_params_groups.push_back(
      torch::optim::OptimizerParamGroup(
          {_xyz}, std::make_unique<torch::optim::AdamOptions>(params.position_lr_init * this->_spatial_lr_scale)));
  optimizer_params_groups.push_back(
      torch::optim::OptimizerParamGroup(
          {_features_dc}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr)));
  optimizer_params_groups.push_back(
      torch::optim::OptimizerParamGroup(
          {_features_rest}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr / 20.)));
  optimizer_params_groups.push_back(
      torch::optim::OptimizerParamGroup(
          {_scaling}, std::make_unique<torch::optim::AdamOptions>(params.scaling_lr * this->_spatial_lr_scale)));
  optimizer_params_groups.push_back(
      torch::optim::OptimizerParamGroup({_rotation}, std::make_unique<torch::optim::AdamOptions>(params.rotation_lr)));
  optimizer_params_groups.push_back(
      torch::optim::OptimizerParamGroup({_opacity}, std::make_unique<torch::optim::AdamOptions>(params.opacity_lr)));

  static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[0].options()).eps(1e-15);
  static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[1].options()).eps(1e-15);
  static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[2].options()).eps(1e-15);
  static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[3].options()).eps(1e-15);
  static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[4].options()).eps(1e-15);
  static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[5].options()).eps(1e-15);

  _optimizer = std::make_unique<torch::optim::Adam>(optimizer_params_groups, torch::optim::AdamOptions(0.f).eps(1e-15));
}

void GaussianModel::prune_optimizer(
    torch::optim::Adam* optimizer,
    const torch::Tensor& mask,
    torch::Tensor& old_tensor,
    int param_position) {
  auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(
      static_cast<torch::optim::AdamParamState&>(*optimizer->state()[c10::guts::to_string(
          optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));
  optimizer->state().erase(
      c10::guts::to_string(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));

  adamParamStates->exp_avg(adamParamStates->exp_avg().index_select(0, mask));
  adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index_select(0, mask));

  optimizer->param_groups()[param_position].params()[0] = old_tensor.index_select(0, mask).set_requires_grad(true);
  old_tensor = optimizer->param_groups()[param_position].params()[0];  // update old tensor
  optimizer
      ->state()[c10::guts::to_string(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] =
      std::move(adamParamStates);
}

void GaussianModel::cat_tensors_to_optimizer(
    torch::optim::Adam* optimizer,
    torch::Tensor& extension_tensor,
    torch::Tensor& old_tensor,
    int param_position) {
  auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(
      static_cast<torch::optim::AdamParamState&>(*optimizer->state()[c10::guts::to_string(
          optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));
  optimizer->state().erase(
      c10::guts::to_string(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));

  adamParamStates->exp_avg(torch::cat({adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)}, 0));
  adamParamStates->exp_avg_sq(torch::cat({adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));

  optimizer->param_groups()[param_position].params()[0] =
      torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
  old_tensor = optimizer->param_groups()[param_position].params()[0];
  old_tensor.retain_grad();
  optimizer
      ->state()[c10::guts::to_string(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] =
      std::move(adamParamStates);
}

std::vector<std::string> GaussianModel::construct_list_of_attributes() {
  std::vector<std::string> attributes = {"x", "y", "z", "nx", "ny", "nz"};

  for (int i = 0; i < _features_dc.size(1) * _features_dc.size(2); ++i)
    attributes.push_back("f_dc_" + std::to_string(i));

  for (int i = 0; i < _features_rest.size(1) * _features_rest.size(2); ++i)
    attributes.push_back("f_rest_" + std::to_string(i));

  attributes.emplace_back("opacity");

  for (int i = 0; i < _scaling.size(1); ++i)
    attributes.push_back("scale_" + std::to_string(i));

  for (int i = 0; i < _rotation.size(1); ++i)
    attributes.push_back("rot_" + std::to_string(i));

  return attributes;
}

void GaussianModel::Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration) {
  // std::cout << "Saving at " << std::to_string(iteration) << " iterations\n";
  auto folder = file_path / ("point_cloud/iteration_" + std::to_string(iteration));
  std::filesystem::create_directories(folder);

  auto xyz = _xyz.cpu().contiguous();
  auto normals = torch::zeros_like(xyz);
  auto f_dc = _features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
  auto f_rest = _features_rest.transpose(1, 2).flatten(1).cpu().contiguous();
  auto opacities = _opacity.cpu();
  auto scale = _scaling.cpu();
  auto rotation = _rotation.cpu();

  std::vector<torch::Tensor> tensor_attributes = {
      xyz.clone(), normals.clone(), f_dc.clone(), f_rest.clone(), opacities.clone(), scale.clone(), rotation.clone()};
  auto attributes = construct_list_of_attributes();
  std::thread t = std::thread([folder, tensor_attributes, attributes]() {
    Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);
  });

  if (isLastIteration) {
    t.join();
  } else {
    t.detach();
  }
}

void GaussianModel::densification_postfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_opacity) {
  cat_tensors_to_optimizer(_optimizer.get(), new_xyz, _xyz, 0);
  cat_tensors_to_optimizer(_optimizer.get(), new_features_dc, _features_dc, 1);
  cat_tensors_to_optimizer(_optimizer.get(), new_features_rest, _features_rest, 2);
  cat_tensors_to_optimizer(_optimizer.get(), new_scaling, _scaling, 3);
  cat_tensors_to_optimizer(_optimizer.get(), new_rotation, _rotation, 4);
  cat_tensors_to_optimizer(_optimizer.get(), new_opacity, _opacity, 5);

  // _xyz_gradient_accum = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
  // _denom = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
  // _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
}

void Write_output_ply(
    const std::filesystem::path& file_path,
    const std::vector<torch::Tensor>& tensors,
    const std::vector<std::string>& attribute_names) {
  tinyply::PlyFile plyFile;

  size_t attribute_offset = 0;  // An offset to track the attribute names

  for (size_t i = 0; i < tensors.size(); ++i) {
    // Calculate the number of columns in the tensor.
    size_t columns = tensors[i].size(1);

    std::vector<std::string> current_attributes;
    for (size_t j = 0; j < columns; ++j) {
      current_attributes.push_back(attribute_names[attribute_offset + j]);
    }

    plyFile.add_properties_to_element(
        "vertex",
        current_attributes,
        tinyply::Type::FLOAT32,
        tensors[i].size(0),
        reinterpret_cast<uint8_t*>(tensors[i].data_ptr<float>()),
        tinyply::Type::INVALID,
        0);

    attribute_offset += columns;  // Increase the offset for the next tensor.
  }

  std::filebuf fb;
  fb.open(file_path, std::ios::out | std::ios::binary);
  std::ostream outputStream(&fb);
  plyFile.write(outputStream, true);  // 'true' for binary format
}
