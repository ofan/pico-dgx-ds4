// CUDA-free production store records. A record owns its complete steering identity, including
// the SSD offsets; lookup and acquisition never silently relabel an old record as current.
#pragma once
#include "model/steer.h"
#include <map>
#include <vector>
#include <utility>
#include <stdexcept>

namespace model::detail {
struct StateRecords {
  struct Sess {
    SteerTag steering;
    std::vector<uint8_t*> pages;    // compressed caches: page i holds positions [i, i + 1) * kPagePos
    int saved = 0;                  // positions [0, saved) are in the pages
    std::map<int, uint8_t*> snaps;  // window snapshots by position, the draft module's windows after them
    uint8_t* end = nullptr;         // every layer's ring and the carries (and the draft module's windows)
    int end_pos = -1;               // at this position
    uint64_t used = 0;              // the clock when an operation last used the session
    // spilled: the same blocks at offsets in the disk tier's file, none in the pool
    bool disk = false;
    std::vector<size_t> dpages;
    std::map<int, size_t> dsnaps;
    size_t dend = SIZE_MAX;
    uint64_t dseq = 0;  // its last write (reading it back waits for that)
    bool kept = false;  // as an earlier run kept it, not read back since: a block that differs is fatal
  };
  // A session in a kept file's index: its blocks' offsets and their checksums as they went out.
  struct Kept {
    SteerTag steering;
    int h = 0, saved = 0, end_pos = -1;
    uint64_t used = 0;
    std::vector<std::pair<size_t, uint64_t>> pages;
    std::map<int, std::pair<size_t, uint64_t>> snaps;
    size_t dend = SIZE_MAX;
    uint64_t dend_sum = 0;
  };
  std::map<int, Sess> sess;

  const Sess* find_current(int h, const SteerTag& current) const {
    auto it = sess.find(h);
    return it != sess.end() && it->second.steering.matches(current) ? &it->second : nullptr;
  }
  Sess& require_current(int h, const SteerTag& current) {
    auto it = sess.find(h);
    if (it == sess.end() || !it->second.steering.matches(current))
      throw std::runtime_error("state: missing or stale steering configuration/epoch");
    return it->second;
  }
  Sess& use_record(int h, const SteerTag& current) {
    auto [it, fresh] = sess.try_emplace(h);
    if (fresh) it->second.steering = current;
    return require_current(h, current);
  }
};
}  // namespace model::detail
