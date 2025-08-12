#include "gp3d/gpmap.h"

namespace GSLIVM {

GpMap::GpMap(GpParameter& param_) : gp_options_{param_} {}

void GpMap::splitPointsIntoCell(PointMatrix& points, std::vector<GSLIVM::GsForLoss>& frameLossPoints) {
  PointMatrix init_points_bucket;
  for (int tmp_i = 0; tmp_i < points.num_point; tmp_i++) {
    Eigen::Vector3d hash_key{
        floor(points.point(0, tmp_i) / gp_options_.grid),
        floor(points.point(1, tmp_i) / gp_options_.grid),
        floor(points.point(2, tmp_i) / gp_options_.grid)};
    std::size_t hash_posi = Vector3DHasher()(hash_key);

    auto [it, inserted] = hash_voxelnode.try_emplace(hash_posi, VoxelNode(hash_posi));
    if (it->second.is_converged) {  // do not add point
      if (frameLossPoints.size() < MAX_SIMI) {
        GSLIVM::GsForLoss _tmp_gs_loss;
        _tmp_gs_loss.hash_posi_ = hash_posi;
        _tmp_gs_loss.gs_xyz =
            torch::tensor({{points.point(0, tmp_i), points.point(1, tmp_i), points.point(2, tmp_i)}}, torch::kFloat32);
        frameLossPoints.push_back(_tmp_gs_loss);
      }
      continue;
    }

    auto [it2, inserted2] = hash_pointmatrix.try_emplace(hash_posi, init_points_bucket);
    if (it2->second.num_point >= 2 * gp_options_.min_points_num_to_gp) {  // do not add point
      continue;
    }
    it2->second.addPoint(points.point.col(tmp_i), gp_options_.variance_sensor);

    updated_voxel.push_back(hash_posi);
    hash_vecpoint.try_emplace(hash_posi, hash_key);
  }
}

void GpMap::updateVariance(std::vector<GSLIVM::varianceUpdate>& update_vas_) {
  updated_voxel.clear();
  for (int tmp_i = 0; tmp_i < update_vas_.size(); tmp_i++) {
    updated_voxel.push_back(update_vas_[tmp_i].hash_);
    hash_voxelnode.find(update_vas_[tmp_i].hash_)->second.is_converged = false;
    Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic>> map_uv(
        update_vas_[tmp_i].update_variance.data(), 1, update_vas_[tmp_i].num_point);
    hash_pointmatrix.find(update_vas_[tmp_i].hash_)->second.variance.block(0, 0, 1, update_vas_[tmp_i].num_point) =
        map_uv;
  }
}

void GpMap::dividePointsIntoCellInitMap(
    GSLIVM::ImageRt& frame,
    bool& is_init,
    Eigen::Matrix<double, 3, Eigen::Dynamic>& points_curr,
    std::vector<GSLIVM::needGPUdata>& updataMapData,
    std::vector<GSLIVM::GsForLoss>& frameLossPoints) {
  if (!is_init) {
    points_notadded = points_curr;
    splitPointsIntoCell(points_notadded, frameLossPoints);
  } else {
    points_scan = points_curr;
    splitPointsIntoCell(points_scan, frameLossPoints);
  }

  std::unordered_map<std::size_t, PointMatrix> voxel_need_processed;
  std::vector<std::size_t> voxel_key;

  for (std::size_t voxel : updated_voxel) {
    if (hash_voxelnode.find(voxel)->second.is_converged) {
      continue;
    }
    if (hash_pointmatrix.find(voxel)->second.num_point == 0) {
      continue;
    }
    // if the number of points is larger than min_points_num_to_gp
    if (hash_pointmatrix.find(voxel)->second.num_point >= gp_options_.min_points_num_to_gp) {
      voxel_need_processed.try_emplace(voxel, hash_pointmatrix.find(voxel)->second);
      voxel_key.push_back(voxel);
    }
  }

  for (int voxel_index = 0; voxel_index < voxel_key.size(); voxel_index++) {
    auto it = voxel_need_processed.find(voxel_key[voxel_index]);
    std::size_t voxel_hash = it->first;

    // region
    auto vecp = hash_vecpoint.find(voxel_hash);
    Eigen::Vector3d hash_index_posi = vecp->second;
    double ix = hash_index_posi.x() * gp_options_.grid;
    double iy = hash_index_posi.y() * gp_options_.grid;
    double iz = hash_index_posi.z() * gp_options_.grid;
    Region region_tmp(ix, iy, iz, ix + gp_options_.grid, iy + gp_options_.grid, iz + gp_options_.grid);

    // reconstruction
    Cell tmp_cell(gp_options_, cam_params_, it->second, region_tmp);

    // add point
    if (tmp_cell.is_surface && tmp_cell.direction != -1) {
      needGPUdata _data;
      _data.point = it->second.point;
      _data.variance = it->second.variance;
      _data.num_point = it->second.num_point;
      _data.region_ = region_tmp;
      _data.hash_ = voxel_hash;
      _data.direction_ = tmp_cell.direction;
      _data.is_converged = hash_voxelnode.find(voxel_hash)->second.is_converged;

      updataMapData.push_back(_data);
      hash_voxelnode.find(voxel_hash)->second.is_converged = true;
    }
  }
}
}  // namespace GSLIVM