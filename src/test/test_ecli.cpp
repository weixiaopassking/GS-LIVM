#include <torch/torch.h>
#include <iostream>

torch::Tensor compute_min_distance(
    const torch::Tensor& points,             // 查询点，形状为 m*3
    const torch::Tensor& spheres_positions,  // 高斯球心位置，形状为 n*3
    const torch::Tensor& scales) {           // 高斯球的尺度，形状为 n

  // 计算球体的平均半径
  auto radius = scales.mean();

  auto m = points.size(0);             // 查询点的数量
  auto n = spheres_positions.size(0);  // 高斯球的数量

  // 扩展查询点和球心位置以便进行向量化计算
  auto points_expanded = points.unsqueeze(1).expand({m, n, 3});
  auto spheres_positions_expanded = spheres_positions.unsqueeze(0).expand({m, n, 3});

  // 计算相对位置
  auto relative_positions = points_expanded - spheres_positions_expanded;

  // 计算查询点到每个球心的距离
  auto distances_to_centers = relative_positions.norm(2, 2);

  // 计算查询点到球表面的距离
  auto distances_to_surface = distances_to_centers - radius;

  // 将负值距离（即点在球内部）设为0
  distances_to_surface = torch::max(distances_to_surface, torch::zeros_like(distances_to_surface));

  // 获取到表面的最小距离
  auto min_distances = std::get<0>(distances_to_surface.min(1));

  return min_distances;
}

int main() {
  int m = 100;  // 查询点数量
  int n = 50;   // 高斯球数量

  // 随机生成查询点和球心位置
  auto points = torch::zeros({m, 3}, torch::device(torch::kCUDA));
  auto spheres_positions = torch::ones({n, 3}, torch::device(torch::kCUDA));

  // 随机生成高斯球的scale
  auto scales = torch::ones({n}, torch::device(torch::kCUDA)) * 0.00001;

  // 计算查询点到高斯球表面的最小距离
  auto min_distances = compute_min_distance(points, spheres_positions, scales);

  std::cout << min_distances[0] << std::endl;

  return 0;
}
