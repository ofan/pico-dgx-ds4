#pragma once
#include <algorithm>
#include <vector>

namespace srv {
// Slot KV and saved snapshots are separate allocations. A live slot must protect its
// session from disk eviction, NOT require its saved copy to remain in device RAM.
// The current command's handles (including CopySnaps' source) do need RAM until
// their operations finish. The caller pins BOTH groups against disk eviction first.
// Explicit spill preserves metadata and slot KV; auto-allocation can then reload
// complete snapshots without competing with redundant copies for all 16 slots.
// With no disk, preserve the original live-slot protection semantics.
template <class Model, class Live>
void spill_other_live_snapshots(Model& m, const std::vector<int>& resident, const Live& live) {
  if (!m.state_disk_capacity()) return;
  for (const auto h : live)
    if (h > 0 && std::find(resident.begin(), resident.end(), h) == resident.end()) m.state_spill(h);
}
}  // namespace srv
