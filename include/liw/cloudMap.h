#pragma once
// c++
#include <math.h>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// opencv
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// pcl
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

// robin_map
#include <tsl/robin_map.h>

template <typename T>
class ThreadSafeQueue {
 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;

 public:
  // 向队列中添加元素
  void push(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
    cond_.notify_one();
  }

  // 获取队列的前端元素
  T front() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      throw std::runtime_error("Queue is empty");
    }
    return queue_.front();
  }

  // 获取队列的后端元素
  T back() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      throw std::runtime_error("Queue is empty");
    }
    return queue_.back();
  }

  // 获取队列的大小
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  // 弹出队列的前端元素
  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    T front = std::move(queue_.front());
    queue_.pop();
    return front;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }
};

extern cv::RNG g_rng;

struct point3D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d raw_point;
  Eigen::Vector3d point;
  Eigen::Vector3d imu_point;

  double intensity;  //   intensity
  int ring;          //   ring
  double timespan;   //   total time of current frame

  double alpha_time = 0.0;
  double relative_time = 0.0;
  double timestamp = 0.0;
  int index_frame = -1;

  point3D() = default;
};

class rgbPoint {
 private:
  Eigen::Vector3f position;
  short rgb[3];
  Eigen::Vector3f cov_rgb;

  double observe_distance;
  double last_observe_time;

 public:
  int point_index;

  short N_rgb;
  short is_out_lier_count;

  Eigen::Vector2d image_velocity;

  rgbPoint(const Eigen::Vector3d& position_);

  void reset();

  void setPosition(const Eigen::Vector3d& position_);

  Eigen::Vector3d getPosition();

  Eigen::Matrix3d getCovRgb();

  Eigen::Vector3d getRgb();

  cv::Scalar getDbgColor();

  pcl::PointXYZI getPositionROS();

  int updateRgb(
      const Eigen::Vector3d& rgb_,
      const double observe_distance_,
      const Eigen::Vector3d observe_sigma_,
      const double observe_time_);
};

struct voxelId {

  voxelId(int kx_, int ky_, int kz_) : kx(kx_), ky(ky_), kz(kz_) {}

  int kx;
  int ky;
  int kz;
};

struct planeParam {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d raw_point;
  Eigen::Vector3d norm_vector;
  Eigen::Matrix<double, 1, 6> jacobians;
  double norm_offset;
  double distance = 0.0;
  double weight = 1.0;

  planeParam() = default;
};

struct imuState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp;

  Eigen::Vector3d un_acc;
  Eigen::Vector3d un_gyr;
  Eigen::Vector3d trans;
  Eigen::Quaterniond quat;
  Eigen::Vector3d vel;

  imuState() = default;
};

struct voxel {

  voxel() = default;

  voxel(short x, short y, short z) : x(x), y(y), z(z) {}

  bool operator==(const voxel& vox) const { return x == vox.x && y == vox.y && z == vox.z; }

  inline bool operator<(const voxel& vox) const {
    return x < vox.x || (x == vox.x && y < vox.y) || (x == vox.x && y == vox.y && z < vox.z);
  }

  inline static voxel coordinates(rgbPoint& point, double voxel_size) {
    return {
        short(point.getPosition().x() / voxel_size),
        short(point.getPosition().y() / voxel_size),
        short(point.getPosition().z() / voxel_size)};
  }

  short x;
  short y;
  short z;
};

struct voxelBlock {

  explicit voxelBlock(int num_points_ = 20) : num_points(num_points_) { points.reserve(num_points_); }

  std::vector<rgbPoint> points;

  double last_visited_time = 0.0;
  bool is_recent = false;

  bool IsFull() const { return num_points == points.size(); }

  void AddPoint(const rgbPoint& point) {
    assert(num_points > points.size());
    points.push_back(point);
  }

  inline int NumPoints() const { return points.size(); }

  inline int Capacity() { return num_points; }

 private:
  int num_points;
};

typedef tsl::robin_map<voxel, voxelBlock> voxelHashMap;

namespace std {

template <>
struct hash<voxel> {
  std::size_t operator()(const voxel& vox) const {
    const size_t kP1 = 73856093;
    const size_t kP2 = 19349669;
    const size_t kP3 = 83492791;
    return vox.x * kP1 + vox.y * kP2 + vox.z * kP3;
  }
};

}  // namespace std