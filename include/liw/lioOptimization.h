#pragma once
// c++
#include <math.h>
#include <fstream>
#include <iostream>
// c++17
#include <filesystem>

#include <queue>
#include <thread>
#include <unordered_set>
#include <vector>

// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

// eigen
#include <Eigen/Core>

// opencv
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

// livox
#include <livox_ros_driver2/CustomMsg.h>

#include "liw/cloudMap.h"

// cloud processing
#include "liw/cloudProcessing.h"

// image processing
#include "liw/imageProcessing.h"

// IMU processing
#include "liw/state.h"

// eskf estimator
#include "liw/eskfEstimator.h"

// utility
#include "liw/parameters.h"
#include "liw/utility.h"

// gs
#include <c10/cuda/CUDACachingAllocator.h>
#include "gs/camera.cuh"
#include "gs/gaussian.cuh"
#include "gs/parameters.cuh"
#include "gs/render_utils.cuh"

// gp
#include "gp3d/gpprocess.cuh"

class imageProcessing;

struct ImageTs {
  cv::Mat image;
  double timestamp;
};

#define ENABLE_PUBLISH false

struct Measurements {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::vector<sensor_msgs::ImuConstPtr> imu_measurements;

  std::pair<double, double> time_sweep;
  std::vector<point3D> lidar_points;

  double time_image = 0.0;
  cv::Mat image;

  bool rendering = false;
};

class cloudFrame {
 public:
  double time_sweep_begin, time_sweep_end;
  double time_frame_begin, time_frame_end;

  int id;      // the index in all_cloud_frame
  int sub_id;  // the index of segment
  int frame_id;

  state* p_state;

  std::vector<point3D> point_frame;

  double offset_begin;
  double offset_end;
  double dt_offset;

  bool success;

  cloudFrame(std::vector<point3D>& point_frame_, state* p_state_);

  cloudFrame(cloudFrame* p_cloud_frame);

  void release();

  cv::Mat rgb_image;
  cv::Mat gray_image;

  int image_cols;
  int image_rows;

  bool if2dPointsAvailable(const double& u, const double& v, const double& scale = 1.0, double fov_mar = -1.0);

  bool getRgb(const double& u, const double& v, int& r, int& g, int& b);

  Eigen::Vector3d
  getRgb(double& u, double& v, int layer, Eigen::Vector3d* rgb_dx = nullptr, Eigen::Vector3d* rgb_dy = nullptr);

  bool project3dTo2d(const pcl::PointXYZI& point_in, double& u, double& v, const double& scale);

  bool project3dPointInThisImage(
      const pcl::PointXYZI& point_in,
      double& u,
      double& v,
      pcl::PointXYZRGB* rgb_point,
      double intrinsic_scale);

  bool project3dPointInThisImage(
      const Eigen::Vector3d& point_in,
      double& u,
      double& v,
      pcl::PointXYZRGB* rgb_point,
      double intrinsic_scale);

  void refreshPoseForProjection();
};

struct Neighborhood {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d center = Eigen::Vector3d::Zero();

  Eigen::Vector3d normal = Eigen::Vector3d::Zero();

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();

  double a2D = 1.0;  // Planarity coefficient
};

struct ResidualBlock {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d point_closest;

  Eigen::Vector3d pt_imu;

  Eigen::Vector3d normal;

  double alpha_time;

  double weight;
};

class estimationSummary {

 public:
  int sample_size = 0;  // The number of points sampled

  int number_of_residuals = 0;  // The number of keypoints used for ICP registration

  int robust_level = 0;

  double distance_correction = 0.0;  // The correction between the last frame's end, and the new frame's beginning

  double relative_distance = 0.0;  // The distance between the beginning of the new frame and the end

  double relative_orientation = 0.0;  // The distance between the beginning of the new frame and the end

  double ego_orientation = 0.0;  // The angular distance between the beginning and the end of the frame

  bool success = true;  // Whether the registration was a success

  std::string error_message;

  estimationSummary();

  void release();
};

struct optimizeSummary {

  bool success = false;  // Whether the registration succeeded

  int num_residuals_used = 0;

  std::string error_log;
};

enum IMAGE_TYPE { RGB8 = 1, COMPRESSED = 2 };

class lioOptimization {
 public:
  bool stop_thread = false;

 private:
  ros::NodeHandle nh;

  ros::Publisher pub_cloud_body;   // the registered cloud of cuurent sweep to be published for visualization
  ros::Publisher pub_cloud_world;  // the cloud of global map to be published for visualization
  ros::Publisher pub_odom;         // the pose of current sweep after LIO-optimization
  ros::Publisher pub_path;         // the position of current sweep after LIO-optimization for visualization
  ros::Publisher pub_cloud_color;
  ros::Publisher pub_cloud_test;

  std::vector<std::shared_ptr<ros::Publisher>> pub_cloud_color_vec;

  ros::Subscriber sub_cloud_ori;  // the data of original point clouds from LiDAR sensor
  ros::Subscriber sub_imu_ori;    // the data of original accelerometer and gyroscope from IMU sensor
  ros::Subscriber sub_img_ori;

  int image_type;

  std::string save_bag_path;
  std::string lidar_topic;
  std::string imu_topic;
  std::string image_topic;

  std::unique_ptr<cloudProcessing> cloud_pro;
  std::unique_ptr<eskfEstimator> eskf_pro;
  std::unique_ptr<imageProcessing> img_pro;
  std::unique_ptr<GaussianModel> gaussian_pro;
  std::unique_ptr<GSLIVM::GpMap> gpmap_pro;
  std::unique_ptr<gpProcess> gpprocess_pro;

  int new_add_point_index = 0;
  int new_gs_points_for_map_count = 0;

  std::vector<GSLIVM::GsForMaps> new_gs_for_map_points;
  std::vector<GSLIVM::GsForLosses> new_gs_for_loss_points;
  std::mutex gs_point_for_map_mutex;
  std::mutex gs_point_for_loss_mutex;
  GSLIVM::GpParameter gp_options_;
  bool extrin_enable;

  double laser_point_cov;

  std::vector<double> v_G;
  std::vector<double> v_extrin_t_imu_lidar;
  std::vector<double> v_extrin_R_imu_lidar;

  Eigen::Matrix3d R_imu_lidar;
  Eigen::Vector3d t_imu_lidar;

  std::vector<double> v_extrin_t_imu_camera;
  std::vector<double> v_extrin_R_imu_camera;

  Eigen::Matrix3d R_imu_camera;
  Eigen::Vector3d t_imu_camera;

  Eigen::Matrix3d R_camera_lidar;
  Eigen::Vector3d t_camera_lidar;
  Eigen::Matrix4d Tcl = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Til = Eigen::Matrix4d::Identity();

  std::vector<double> v_camera_intrinsic;
  std::vector<double> v_camera_dist_coeffs;

  Eigen::Vector3d pose_lid;

  ThreadSafeQueue<ImageTs> time_img_buffer;

  ThreadSafeQueue<sensor_msgs::Imu::ConstPtr> imu_buffer;
  ThreadSafeQueue<point3D> point_buffer;

  std::vector<cloudFrame*> all_cloud_frame;

  std::vector<std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>>> nearest_points;

  double start_time_img;

  double last_time_lidar;
  double last_time_imu;
  double last_time_img;

  double last_get_measurement;
  bool last_rendering;
  double last_time_frame;
  double current_time;

  int index_frame;

  double time_max_solver;
  int num_max_iteration;

  voxelHashMap voxel_map;
  voxelHashMap color_voxel_map;
  Hash_map_3d<long, rgbPoint*> hashmap_3d_points;

  odometryOptions odometry_options;
  mapOptions map_options;

  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;
  nav_msgs::Path path;

  double dt_sum;

  std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_meas;
  std::vector<imuState> imu_states;

  std::vector<voxelId> voxels_recent_visited_temp;

  // add
  Eigen::Quaterniond last_image_rotation;
  Eigen::Vector3d last_image_trans;
  double last_image_time;

  int image_frame_id = 0;
  // gs fusion
  std::vector<std::vector<Camera>> _cameras;
  // std::vector<CameraInfo> _cameraInfos;

  bool is_gs_started = false;
  bool is_received_data = false;

  // gaussian-splatting
  gs::param::ModelParameters gsmodelParams;
  gs::param::OptimizationParameters gsoptimParams;

  int iter = 0;
  const int window_size = 11;
  const int channel = 3;
  torch::Tensor conv_window;
  torch::Tensor background;
  int last_status_len = 0;
  std::vector<int> selected_indices_hist;
  std::vector<int> selected_indices_curr;
  float avg_converging_rate = 0.f;
  int image_filter_num;
  int image_filter_index = 0;

  int updateVas_size_iter_debug = 0;

  int image_width_verify;
  int image_height_verify;

  ros::Timer check_timer;

 public:
  // initialize class
  lioOptimization();

  void gsPointCloudUpdate(
      cloudFrame*& p_frame,
      int& updated_voxel_count,
      GSLIVM::GsForMaps& final_gs_sample,
      GSLIVM::GsForLosses& final_gs_calc_loss);

  void readParameters();

  void allocateMemory();

  void initialValue();
  // initialize class

  // get sensor data
  void livoxHandler(const livox_ros_driver2::CustomMsg::ConstPtr& msg);

  void standardCloudHandler(const sensor_msgs::PointCloud2::ConstPtr& msg);

  void imuHandler(const sensor_msgs::Imu::ConstPtr& msg);

  void imageHandler(const sensor_msgs::ImageConstPtr& msg);

  void heartHandler(const ros::TimerEvent& event);
  void compressedImageHandler(const sensor_msgs::CompressedImageConstPtr& msg);
  // get sensor data

  // main loop
  std::vector<Measurements> getMeasurements();
  bool compareStatesImageAdd(
      const Eigen::Quaterniond& R1,
      const Eigen::Vector3d& t1,
      double& time_1,
      const Eigen::Quaterniond& R2,
      const Eigen::Vector3d& t2,
      double& time_2);

  void process(
      std::vector<point3D>& cut_sweep,
      double timestamp_begin,
      double timestamp_offset,
      cv::Mat& cur_image,
      bool to_rendering);

  void run();
  // main loop

  // data handle and state estimation
  cloudFrame*
  buildFrame(std::vector<point3D>& const_frame, state* cur_state, double timestamp_begin, double timestamp_offset);

  void makePointTimestamp(std::vector<point3D>& sweep, double time_begin, double time_end);

  void stateInitialization(state* cur_state);

  optimizeSummary stateEstimation(cloudFrame* p_frame, bool to_rendering);

  optimizeSummary optimize(cloudFrame* p_frame, const icpOptions& cur_icp_options, double sample_voxel_size);

  optimizeSummary buildPlaneResiduals(
      const icpOptions& cur_icp_options,
      voxelHashMap& voxel_map_temp,
      std::vector<point3D>& keypoints,
      std::vector<planeParam>& plane_residuals,
      cloudFrame* p_frame,
      double& loss_sum);

  optimizeSummary updateIEKF(
      const icpOptions& cur_icp_options,
      voxelHashMap& voxel_map_temp,
      std::vector<point3D>& keypoints,
      cloudFrame* p_frame);

  Neighborhood computeNeighborhoodDistribution(
      const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> searchNeighbors(
      voxelHashMap& map,
      const Eigen::Vector3d& point,
      int nb_voxels_visited,
      double size_voxel_map,
      int max_num_neighbors,
      int threshold_voxel_capacity = 1,
      std::vector<voxel>* voxels = nullptr);
  // data handle and state estimation

  // map update
  void addPointToMap(
      voxelHashMap& map,
      rgbPoint& point,
      double voxel_size,
      int max_num_points_in_voxel,
      double min_distance_points,
      int min_num_points,
      cloudFrame* p_frame);

  void addPointToColorMap(
      voxelHashMap& map,
      rgbPoint& point,
      double voxel_size,
      int max_num_points_in_voxel,
      double min_distance_points,
      int min_num_points,
      cloudFrame* p_frame,
      std::vector<voxelId>& voxels_recent_visited_temp);

  void addPointsToMap(
      voxelHashMap& map,
      cloudFrame* p_frame,
      double voxel_size,
      int max_num_points_in_voxel,
      double min_distance_points,
      int min_num_points = 0,
      bool to_rendering = false);

  void removePointsFarFromLocation(voxelHashMap& map, const Eigen::Vector3d& location, double distance);

  size_t mapSize(const voxelHashMap& map);
  // map update

  // save result to device
  void recordSinglePose(cloudFrame* p_frame);
  // save result to device

  // publish result by ROS for visualization
  void publish_path(ros::Publisher pub_path, cloudFrame* p_frame);

  void set_posestamp(geometry_msgs::PoseStamped& body_pose_out, cloudFrame* p_frame);

  void addPointToPcl(pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_points, rgbPoint& point, cloudFrame* p_frame);

  void publishCLoudWorld(
      ros::Publisher& pub_cloud_world,
      pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullRes,
      cloudFrame* p_frame);

  pcl::PointCloud<pcl::PointXYZI>::Ptr points_world;
  std::mutex color_points_mutex;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_points_world;
  // void publish_odometry(const ros::Publisher& pubOdomAftMapped, cloudFrame* p_frame);

  void pubColorPoints(ros::Publisher& pub_cloud_rgb, cloudFrame* p_frame);

  // publish result by ROS for visualization
  void threadAddColorPoints();

  void saveColorPoints();
  void gsAddCamera(cloudFrame* p_frame, std::vector<Camera>& cams);

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform laserOdometryTrans;

  cv::Mat tensor2CvMat2X(torch::Tensor& tensor, float maxDepth = 50.0f);
  cv::Mat tensor2CvMat3X(torch::Tensor& tensor);

  void saveDepthMapAsNPY(torch::Tensor& tensor, const std::string& filename);

  float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img);

  void saveRender();

  // void optimize_vis(int updated_voxel_count, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& gp_points_colors_to_loss);
  void optimize_vis();

  std::pair<std::vector<int>, std::vector<int>> get_random_indices(
      int max_size,
      std::vector<int> exist_curr,
      std::vector<int> exist_hist,
      int window_size,
      int curr_size,
      int hist_size);

  std::vector<int> findAndErase(std::vector<int>& vec1, const std::vector<int>& vec2);

  void processAndMergePointClouds(GSLIVM::GsForMaps& all_gs);
  void processAndMergeLosses(GSLIVM::GsForLosses& all_gs_losses);
};
