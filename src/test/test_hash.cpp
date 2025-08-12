#include <gp3d/gpmap.h>
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <unordered_set>

void printProgress(double progress) {
  int val = static_cast<int>(progress * 100);
  std::cout << "\r[";
  for (int i = 0; i < 50; ++i) {
    if (i < val / 2)
      std::cout << "=";
    else
      std::cout << " ";
  }
  std::cout << "] " << std::setw(3) << val << " %  " << std::flush;
}

int main() {
  // 设置遍历范围和间隔
  double STEP = 20;
  double xmin = -STEP, xmax = STEP;
  double ymin = -STEP, ymax = STEP;
  double zmin = -STEP, zmax = STEP;
  double interval = 0.2;

  // 使用unordered_set提高查找效率
  std::unordered_set<std::size_t> hash_set;

  // 遍历三维空间
  long total_points = ((xmax - xmin) / interval + 1) * ((ymax - ymin) / interval + 1) * ((zmax - zmin) / interval + 1);
  long current_point = 0;

  for (double x = xmin; x <= xmax; x += interval) {
    for (double y = ymin; y <= ymax; y += interval) {
      for (double z = zmin; z <= zmax; z += interval) {
        Eigen::Vector3d point(x, y, z);
        std::size_t hash_value = GSLIVOM::Vector3DHasher()(point);
        if (hash_set.find(hash_value) != hash_set.end()) {
          std::cout << "\nhash crash: " << hash_value << " at (" << x << ", " << y << ", " << z << ")" << std::endl;
        } else {
          hash_set.insert(hash_value);
        }
        current_point++;
        printProgress(static_cast<double>(current_point) / total_points);
      }
    }
  }
  std::cout << "\nDone." << std::endl;

  return 0;
}