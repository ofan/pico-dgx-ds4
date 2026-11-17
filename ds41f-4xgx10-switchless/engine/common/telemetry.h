// Numeric-only telemetry sink shared by model and server. No CUDA, I/O, or locks.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
namespace pico::metrics {
static_assert(std::atomic<uint64_t>::is_always_lock_free, "telemetry requires lock-free 64-bit atomics");
using Atomic = std::atomic<uint64_t>;
inline uint64_t wall_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
inline uint64_t steady_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline uint64_t read(const Atomic& x) { return x.load(std::memory_order_relaxed); }
inline void set(Atomic& x, uint64_t n) { x.store(n, std::memory_order_relaxed); }
inline void add(Atomic& x, uint64_t n=1) { x.fetch_add(n, std::memory_order_relaxed); }
inline void sub(Atomic& x, uint64_t n=1) { x.fetch_sub(n, std::memory_order_relaxed); }
inline constexpr std::array<uint64_t, 19> bounds_us = {
  1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,
  2000000,5000000,10000000,20000000,30000000,60000000,120000000,300000000,600000000};
struct alignas(64) Histogram {
  std::array<Atomic, bounds_us.size()+1> bins{};
  Atomic sum_us{0};
  void observe(uint64_t us) {
    size_t b=0; while (b<bounds_us.size() && us>bounds_us[b]) ++b;
    add(sum_us, us); add(bins[b]);
  }
};
enum Phase : size_t { Scheduler, Prefill, Decode };
struct Registry {
  // Independently written producer groups. All storage is bounded at compile time.
  alignas(64) Atomic waiting{0}, received{0};
  alignas(64) Atomic running{0}, prompt{0}, generated{0}, reused{0}, prefilled{0};
  Atomic slots{0}, slots_capacity{0}, context_capacity{0};
  std::array<Atomic,6> outcomes{}; // stop,length,context,cancelled,error,unknown
  Histogram ttft, tpot, queue;
  alignas(64) std::array<Atomic,3> progress{};
  alignas(64) Atomic memory_bytes{0}, memory_capacity{0}, ssd_bytes{0}, ssd_capacity{0}, sessions{0};
  Atomic state_observed{0};
  alignas(64) Atomic initialization{0}; // 0 loading, 1 ready, 2 stopping
};
inline Registry registry;
static_assert(sizeof(Registry) < (1<<20), "registry budget");
// Per-request ownership stays on the scheduler thread, never on the scrape thread.
// Exposed here so terminal / cancellation / resume accounting is CPU-testable.
struct RequestMetrics {
  bool started = false, finished = false;
  uint64_t received_us = 0, first_us = 0, outputs = 0;
  void receive(uint64_t now) {
    received_us = now;
    add(registry.received); add(registry.waiting);
  }
  void start(uint64_t now) {
    if (started || finished) return;
    started = true; sub(registry.waiting); add(registry.running);
    registry.queue.observe(now >= received_us ? now - received_us : 0);
  }
  void token(uint64_t now) {
    if (finished) return;
    if (outputs++ == 0) {
      first_us = now;
      registry.ttft.observe(now >= received_us ? now - received_us : 0);
    }
    add(registry.generated);
  }
  void finish(size_t outcome, uint64_t now) {
    if (finished) return;
    finished = true;
    if (started) sub(registry.running); else sub(registry.waiting);
    add(registry.outcomes[outcome < registry.outcomes.size() ? outcome : 5]);
    if (outputs > 1) registry.tpot.observe((now >= first_us ? now - first_us : 0) / (outputs - 1));
  }
};
inline void progress(Phase p) { set(registry.progress[p], wall_us()); }
} // namespace pico::metrics
