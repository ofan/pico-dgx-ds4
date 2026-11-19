// CUDA-free command pipeline shared by the server, all ranks and the CPU regression.
// Only the model's storage/device effects and the broadcast transport are substituted in CPU tests.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "server/snapshot_policy.h"

namespace srv {
// A session's state operation (model.h state_*), applied by every rank in order.
enum : int32_t {
  kStSave = 1,     // state_save(h, a, b)
  kStSnapEnd,      // state_snap_end(h, a)
  kStLoad,         // state_load(h, a)
  kStRestore,      // state_restore(h, a)
  kStRestoreEnd,   // state_restore_end(h)
  kStDrop,         // state_drop(h, a)
  kStFree,         // state_free(h)
  kStCopySnaps,    // state_copy_snaps(h, a, b): h's snapshots at <= b into session a
  kStDropSnap,     // state_drop_snap(h, a)
  kStSnap,         // state_snap(h, a)
  kStSpill,        // state_spill(h)
  kStUnspill,      // state_unspill(h)
  kStKeep,         // state_keep(gen): a planned stop's kept sessions (gen = a | b << 32)
};
struct StateOp {
  int32_t op = 0, h = -1, a = 0, b = 0;
};

constexpr int kMaxOps = 12;
constexpr int32_t kCmdState = 5;       // ordinary state operations, with pin/spill scheduling
constexpr int32_t kCmdEpochFree = 23;  // epoch teardown: free only, never acquire/spill/retain old records

// Every rank uses this path, including followers before dispatch. Old records can legitimately have
// different epochs after a partial apply. Destruction must not acquire them under the current tag.
template <class Model, class Command, class Apply>
void apply_state_command(Model& m, const Command& c, Apply apply) {
  if (c.op == kCmdEpochFree) {
    // Validate the entire batch before changing anything; this mode cannot bypass guards for reads,
    // writes, spill or keep. Drop old eviction pins, but do not select a slot or run scheduling.
    if (c.n_ops < 0 || c.n_ops > kMaxOps) throw std::runtime_error("epoch teardown: invalid batch size");
    for (int i = 0; i < c.n_ops; ++i)
      if (c.ops[i].op != kStFree || c.ops[i].h <= 0)
        throw std::runtime_error("epoch teardown: only session frees are allowed");
    m.state_pin({});
    for (int i = 0; i < c.n_ops; ++i) m.state_free(c.ops[i].h);
    return;
  }
  if (c.slot >= 0) m.use_slot(c.slot);
  std::vector<int> pin;
  for (int i = 0; i < c.n_ops; ++i) {
    pin.push_back(c.ops[i].h);
    if (c.ops[i].op == kStCopySnaps) pin.push_back(c.ops[i].a);
  }
  if (c.handle >= 0) pin.push_back(c.handle);
  const std::vector<int> resident = pin;
  for (const int32_t h : c.live)
    if (h > 0) pin.push_back(h);
  m.state_pin(pin);
  spill_other_live_snapshots(m, resident, c.live);
  for (int i = 0; i < c.n_ops; ++i) apply(m, c.ops[i]);
}

// Rank zero's real issue path: live metadata remains present during teardown. Correctness must come
// from the command mode on EVERY rank, not from clearing rank-zero bookkeeping early.
template <class Command, class Sessions, class Slots, class Broadcast, class Apply>
void issue_state_command(Command& c, const Sessions& sessions, const Slots& slots,
                         Broadcast broadcast, Apply apply) {
  for (size_t s = 0; s < slots.size(); ++s) {
    const auto it = slots[s] ? sessions.find(slots[s]) : sessions.end();
    c.live[s] = it == sessions.end() ? 0 : it->second.h;
  }
  broadcast(c);
  apply(c);
}

template <class Command, class Issue>
void state_command_batches(const std::vector<StateOp>& v, int32_t op, Issue issue) {
  for (size_t i = 0; i < v.size(); i += kMaxOps) {
    Command c{};
    c.op = op;
    c.n_ops = (int)std::min<size_t>(kMaxOps, v.size() - i);
    std::copy(v.begin() + i, v.begin() + i + c.n_ops, c.ops);
    issue(c);
  }
}

// The production clear callback (both successful transitions and every recovery attempt).
template <class Command, class Model, class Sessions, class Slots, class Later, class Issue>
size_t clear_retained_state(Model& m, Sessions& sessions, Slots& slots, Later& later, Issue issue) {
  std::vector<StateOp> v;
  for (const int h : m.state_sessions()) v.push_back({kStFree, h, 0, 0});
  state_command_batches<Command>(v, kCmdEpochFree, issue);
  sessions.clear();
  for (auto& s : slots) s = 0;
  later.clear();
  return v.size();
}
}  // namespace srv
