#include "liw/cloudProcessing.h"
#include "liw/utility.h"

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

cloudProcessing::cloudProcessing() {
  point_filter_num = 1;
  last_end_time = -1;
  sweep_id = 0;
}

void cloudProcessing::setLidarType(int para) {
  lidar_type = para;
}

void cloudProcessing::setNumScans(int para) {
  N_SCANS = para;

  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<pcl::PointXYZINormal> v_cloud_temp;
    v_cloud_temp.clear();
    scan_cloud.push_back(v_cloud_temp);
  }

  assert(N_SCANS == scan_cloud.size());

  for (int i = 0; i < N_SCANS; i++) {
    std::vector<extraElement> v_elem_temp;
    v_extra_elem.push_back(v_elem_temp);
  }

  assert(N_SCANS == v_extra_elem.size());
}

void cloudProcessing::setScanRate(int para) {
  SCAN_RATE = para;
  time_interval_sweep = 1 / double(SCAN_RATE);
}

void cloudProcessing::setTimeUnit(int para) {
  time_unit = para;

  switch (time_unit) {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }
}

void cloudProcessing::setBlind(double para) {
  blind = para;
}

void cloudProcessing::setDetRange(double para) {
  det_range = para;
}

void cloudProcessing::setExtrinR(Eigen::Matrix3d& R) {
  R_imu_lidar = R;
}

void cloudProcessing::setExtrinT(Eigen::Vector3d& t) {
  t_imu_lidar = t;
}

void cloudProcessing::setPointFilterNum(int para) {
  point_filter_num = para;
}

void cloudProcessing::printfFieldName(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  std::cout << "Input pointcloud field names: [" << msg->fields.size() << "]: ";

  for (int i = 0; i < msg->fields.size(); i++) {
    std::cout << msg->fields[i].name << ", ";
  }

  std::cout << std::endl;
}

void cloudProcessing::process(const sensor_msgs::PointCloud2::ConstPtr& msg, ThreadSafeQueue<point3D>& point_buffer) {
  switch (lidar_type) {
    case OUST:
      ousterHandler(msg, point_buffer);
      break;

    case VELO:
      velodyneHandler(msg, point_buffer);
      break;

    case ROBO:
      robosenseHandler(msg, point_buffer);
      break;

    case PANDAR:
      pandarHandler(msg, point_buffer);
      break;

    default:
      ROS_ERROR("Only Velodyne LiDAR interface is supported currently.");
      printfFieldName(msg);
      break;
  }

  sweep_id++;
}

void cloudProcessing::livoxHandler(
    const livox_ros_driver2::CustomMsg::ConstPtr& msg,
    ThreadSafeQueue<point3D>& point_buffer) {
  int plsize = msg->point_num;
  static double tm_scale = 1e9;

  double headertime = msg->header.stamp.toSec();
  double timespan_ = msg->points.back().offset_time / tm_scale;

  for (int i = 0; i < plsize; i++) {
    if (!(std::isfinite(msg->points[i].x) && std::isfinite(msg->points[i].y) && std::isfinite(msg->points[i].z)))
      continue;

    if (i % point_filter_num != 0)
      continue;

    double range = sqrt(
        msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y +
        msg->points[i].z * msg->points[i].z);
    if (range > det_range || range < blind)
      continue;

    if (/*(msg->points[i].line < N_SCANS) &&*/ (
        (msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
      point3D point_temp;
      point_temp.raw_point = Eigen::Vector3d(msg->points[i].x, msg->points[i].y, msg->points[i].z);
      point_temp.point = point_temp.raw_point;
      point_temp.relative_time = msg->points[i].offset_time / tm_scale;  // curvature unit: ms
      point_temp.intensity = msg->points[i].reflectivity;

      point_temp.timestamp = headertime + point_temp.relative_time;
      point_temp.alpha_time = point_temp.relative_time / timespan_;
      point_temp.timespan = timespan_;
      point_temp.ring = msg->points[i].line;

      point_buffer.push(point_temp);
    }
  }
}

void cloudProcessing::velodyneHandler(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    ThreadSafeQueue<point3D>& point_buffer) {

  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();

  double headertime = msg->header.stamp.toSec();

  static double tm_scale = 1;

  auto time_list_velodyne = [&](velodyne_ros::Point& point_1, velodyne_ros::Point& point_2) {
    return (point_1.time < point_2.time);
  };
  sort(pl_orig.points.begin(), pl_orig.points.end(), time_list_velodyne);
  while (pl_orig.points[plsize - 1].time / tm_scale >= 0.1) {
    plsize--;
    pl_orig.points.pop_back();
  }
  double timespan_ = pl_orig.points.back().time / tm_scale;

  for (int i = 0; i < plsize; i++) {
    if (!(std::isfinite(pl_orig.points[i].x) && std::isfinite(pl_orig.points[i].y) &&
          std::isfinite(pl_orig.points[i].z)))
      continue;

    if (i % point_filter_num != 0)
      continue;

    double range = sqrt(
        pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
        pl_orig.points[i].z * pl_orig.points[i].z);
    if (range > det_range || range < blind)
      continue;

    point3D point_temp;
    point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
    point_temp.point = point_temp.raw_point;
    point_temp.relative_time = pl_orig.points[i].time / tm_scale;  // curvature unit: s
    point_temp.intensity = pl_orig.points[i].intensity;

    point_temp.timestamp = headertime + point_temp.relative_time;
    point_temp.alpha_time = point_temp.relative_time / timespan_;
    point_temp.timespan = timespan_;
    point_temp.ring = pl_orig.points[i].ring;

    if (last_end_time != -1 && abs(point_temp.timestamp - last_end_time) > 1e3) {
      continue;
    }

    point_buffer.push(point_temp);
    last_end_time = point_temp.timestamp;
  }
}

void cloudProcessing::ousterHandler(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    ThreadSafeQueue<point3D>& point_buffer) {
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);

  static double tm_scale = 1e9;

  double headertime = msg->header.stamp.toSec();
  double timespan_ = pl_orig.points.back().t / tm_scale;

  for (int i = 0; i < pl_orig.points.size(); i++) {
    if (!(std::isfinite(pl_orig.points[i].x) && std::isfinite(pl_orig.points[i].y) &&
          std::isfinite(pl_orig.points[i].z)))
      continue;

    if (i % point_filter_num != 0)
      continue;

    double range = sqrt(
        pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
        pl_orig.points[i].z * pl_orig.points[i].z);
    if (range > det_range || range < blind)
      continue;

    point3D point_temp;
    point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
    point_temp.point = point_temp.raw_point;
    point_temp.relative_time = pl_orig.points[i].t / tm_scale;
    point_temp.intensity = pl_orig.points[i].intensity;

    point_temp.timestamp = headertime + point_temp.relative_time;
    point_temp.alpha_time = point_temp.relative_time / timespan_;
    point_temp.timespan = timespan_;
    point_temp.ring = pl_orig.points[i].ring;

    if (last_end_time != -1 && abs(point_temp.timestamp - last_end_time) > 1e3) {
      continue;
    }
    point_buffer.push(point_temp);
    last_end_time = point_temp.timestamp;
  }
}

void cloudProcessing::robosenseHandler(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    ThreadSafeQueue<point3D>& point_buffer) {
  pcl::PointCloud<robosense_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();

  double headertime = msg->header.stamp.toSec();
  auto time_list_robosense = [&](robosense_ros::Point& point_1, robosense_ros::Point& point_2) {
    return (point_1.timestamp < point_2.timestamp);
  };

  sort(pl_orig.points.begin(), pl_orig.points.end(), time_list_robosense);

  while (pl_orig.points[plsize - 1].timestamp - pl_orig.points[0].timestamp >= 0.1) {
    plsize--;
    pl_orig.points.pop_back();
  }

  double timespan_ = pl_orig.points.back().timestamp - pl_orig.points[0].timestamp;

  for (int i = 0; i < pl_orig.points.size(); i++) {
    if (!(std::isfinite(pl_orig.points[i].x) && std::isfinite(pl_orig.points[i].y) &&
          std::isfinite(pl_orig.points[i].z)))
      continue;

    double range = sqrt(
        pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
        pl_orig.points[i].z * pl_orig.points[i].z);
    if (range > det_range || range < blind)
      continue;

    point3D point_temp;
    point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
    point_temp.point = point_temp.raw_point;
    point_temp.relative_time = pl_orig.points[i].timestamp - pl_orig.points[0].timestamp;  // curvature unit: s
    point_temp.intensity = (double)pl_orig.points[i].intensity;

    point_temp.timestamp = pl_orig.points[i].timestamp;
    point_temp.alpha_time = point_temp.relative_time / timespan_;
    point_temp.timespan = timespan_;
    point_temp.ring = pl_orig.points[i].ring;
    if (point_temp.alpha_time > 1 || point_temp.alpha_time < 0)
      std::cout << point_temp.alpha_time << ", this may error." << std::endl;

    if (last_end_time != -1 && abs(point_temp.timestamp - last_end_time) > 1e3) {
      continue;
    }

    point_buffer.push(point_temp);
    last_end_time = point_temp.timestamp;
  }
}

void cloudProcessing::pandarHandler(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    ThreadSafeQueue<point3D>& point_buffer) {
  pcl::PointCloud<pandar_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();

  double headertime = msg->header.stamp.toSec();

  static double tm_scale = 1;  //   1e6

  auto time_list_pandar = [&](pandar_ros::Point& point_1, pandar_ros::Point& point_2) {
    return (point_1.timestamp < point_2.timestamp);
  };
  std::sort(pl_orig.points.begin(), pl_orig.points.end(), time_list_pandar);
  while (pl_orig.points[plsize - 1].timestamp - pl_orig.points[0].timestamp >= 0.1) {
    plsize--;
    pl_orig.points.pop_back();
  }
  double timespan_ = pl_orig.points.back().timestamp - pl_orig.points[0].timestamp;

  for (int i = 0; i < plsize; i++) {
    if (!(std::isfinite(pl_orig.points[i].x) && std::isfinite(pl_orig.points[i].y) &&
          std::isfinite(pl_orig.points[i].z)))
      continue;

    if (i % point_filter_num != 0)
      continue;

    double range = sqrt(
        pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
        pl_orig.points[i].z * pl_orig.points[i].z);

    if (range > det_range || range < blind) {
      continue;
    }

    point3D point_temp;
    point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
    point_temp.point = point_temp.raw_point;
    point_temp.relative_time = pl_orig.points[i].timestamp - pl_orig.points[0].timestamp;
    point_temp.intensity = pl_orig.points[i].intensity;

    point_temp.timestamp = headertime + point_temp.relative_time;
    point_temp.alpha_time = point_temp.relative_time / timespan_;
    point_temp.timespan = timespan_;
    point_temp.ring = pl_orig.points[i].ring;

    if (last_end_time != -1 && abs(point_temp.timestamp - last_end_time) > 1e3) {
      continue;
    }

    point_buffer.push(point_temp);
    last_end_time = point_temp.timestamp;
  }
}