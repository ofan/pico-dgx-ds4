#pragma once
#include "common/telemetry.h"
#include <atomic>
#include <string>
#include <thread>
namespace pico::metrics {
// Configured once before the listener starts. The caller retains ownership until workers stop.
class Endpoint {
 public:
  Endpoint(std::string address, std::string model_name, int rank, int cpu=-1);
  ~Endpoint();
  Endpoint(const Endpoint&)=delete;
  Endpoint& operator=(const Endpoint&)=delete;
  std::string render() const;
  int port() const { return port_; }
 private:
  void run();
  std::string labels_;
  uint64_t started_;
  Atomic heartbeat_{0};
  int fd_=-1, port_=0, cpu_=-1, rank_=0;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};
} // namespace pico::metrics
