// #include "common/timer/timer.h"
#include "common/timer/timer.h"

#include <glog/logging.h>
#include <fstream>
#include <numeric>

namespace common {

std::map<std::string, Timer::TimerRecord> Timer::records_;

void Timer::DumpIntoFile(const int camera_size, const double duration, const std::string& file_name) {
  std::ofstream ofs(file_name, std::ios::out);
  if (!ofs.is_open()) {
    std::cout << "Failed to open file: " << file_name << std::endl;
    return;
  } else {
    std::cout << "Dump Time Records into file: " + file_name << std::endl;
  }

  double realtime_ms = duration / camera_size;
  ofs << std::fixed << std::setprecision(9) << realtime_ms << std::endl;

  size_t max_length = 0;
  for (const auto& iter : records_) {
    ofs << iter.first << ", ";
    if (iter.second.time_usage_in_ms_.size() > max_length) {
      max_length = iter.second.time_usage_in_ms_.size();
    }
  }
  ofs << std::endl;

  for (size_t i = 0; i < max_length; ++i) {
    for (const auto& iter : records_) {
      if (i < iter.second.time_usage_in_ms_.size()) {
        ofs << std::fixed << std::setprecision(15) << iter.second.time_usage_in_ms_[i].second << "="
            << iter.second.time_usage_in_ms_[i].first << ",";
      } else {
        ofs << ",";
      }
    }
    ofs << std::endl;
  }
  ofs.close();
}

}  // namespace common