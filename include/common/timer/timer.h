#ifndef FUSION_TIMER_H
#define FUSION_TIMER_H
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace common {

class Timer {
 public:
  struct TimerRecord {
    TimerRecord() = default;

    TimerRecord(const std::string& name, double time_usage, double time_stamp) {
      func_name_ = name;
      time_usage_in_ms_.emplace_back(std::make_pair(time_usage, time_stamp));
    }

    std::string func_name_;
    std::vector<std::pair<double, double>> time_usage_in_ms_;
  };

  /**
     * record
     * @tparam F
     * @param func
     * @param func_name
     */
  template <class F>
  static void Evaluate(bool log_time, const double time_stamp, F&& func, const std::string& func_name) {
    auto t1 = std::chrono::steady_clock::now();
    std::forward<F>(func)();
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;

    if (log_time) {
      std::cout << "time_stamp: " << time_stamp << " --> " << func_name << ": " << time_used << " ms" << std::endl;
    }

    if (records_.find(func_name) != records_.end()) {
      records_[func_name].time_usage_in_ms_.emplace_back(std::make_pair(time_used, time_stamp));
    } else {
      records_.insert({func_name, TimerRecord(func_name, time_used, time_stamp)});
    }
  }

  static void DumpIntoFile(const int camera_size, const double duration, const std::string& file_name);

  static void Clear() { records_.clear(); }

 private:
  static std::map<std::string, TimerRecord> records_;
};

}  // namespace common

#endif  // FUSION_TIMER_H
