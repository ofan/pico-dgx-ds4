// CPU checks for production steering validation, hashing, four-rank coordination, device transaction
// and cache-record enforcement. Device/transport stand-ins are not GPU or physical SSD qualification.
#include "model/steer.h"
#include "model/state_records.h"
#include "server/state_commands.h"
#include "model/config.h"
#include <map>
#include <set>
#include <stdexcept>
#include <array>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace model;

static int g_fail = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fail;
  }
}

static SteerConfig cfg(bool on, float alpha, uint64_t layers, uint64_t vector_id) {
  SteerConfig c;
  c.on = on, c.alpha = alpha, c.layers = layers, c.vector_id = vector_id;
  return c;
}

static void test_parse() {
  float a = -1.f;
  std::string why;
  check(steer_parse_alpha("0.75", &a, &why) && a == 0.75f, "0.75 parses");
  check(steer_parse_alpha("0", &a, &why) && a == 0.f, "0 parses");
  check(steer_parse_alpha("2", &a, &why) && a == 2.f, "the upper bound parses");
  check(steer_parse_alpha("1.5", &a, &why) && a == 1.5f, "1.5 parses");
  check(!steer_parse_alpha("2.0001", &a, &why), "above the range is refused");
  check(!steer_parse_alpha("-0.1", &a, &why), "below the range is refused");
  check(!steer_parse_alpha("nan", &a, &why), "nan is refused");
  check(!steer_parse_alpha("inf", &a, &why), "inf is refused");
  check(!steer_parse_alpha("0.75x", &a, &why), "trailing text is refused");
  check(!steer_parse_alpha(" 0.75", &a, &why), "leading space is refused");
  check(!steer_parse_alpha("", &a, &why), "empty is refused");
  check(!steer_parse_alpha("0.75 ", &a, &why), "trailing space is refused");
  check(!steer_parse_alpha("1e400", &a, &why), "overflow to inf is refused");
}

static void test_check() {
  const SteerConfig on = cfg(true, 0.75f, 0x3full, 9);
  check(steer_check(on, 0.75f) == SteerCheck::kNoop, "the current value is a no-op");
  check(steer_check(on, 1.0f) == SteerCheck::kApply, "a different value applies");
  check(steer_check(on, 0.0f) == SteerCheck::kApply, "zero applies when the current value is not zero");
  check(steer_check(on, -0.0f) == SteerCheck::kApply, "-0.0 is not the same value as +0.0");
  check(steer_check(on, 2.5f) == SteerCheck::kBadAlpha, "out of range");
  check(steer_check(on, NAN) == SteerCheck::kBadAlpha, "nan");
  check(steer_check(on, INFINITY) == SteerCheck::kBadAlpha, "inf");
  const SteerConfig off = cfg(false, 0.f, 0, 0);
  check(steer_check(off, 0.75f) == SteerCheck::kSteeringOff, "alpha does not enable steering");
  check(steer_check(off, 5.f) == SteerCheck::kBadAlpha, "range is checked before the on/off state");
}

static void test_identity() {
  const SteerConfig a = cfg(true, 0.75f, 0x3full, 9);
  const SteerConfig b = cfg(true, 0.75f, 0x3full, 9);
  check(a == b && steer_config_id(a) == steer_config_id(b), "equal configurations have equal ids");
  check(steer_config_id(a) != steer_config_id(cfg(true, 0.7501f, 0x3full, 9)), "alpha is in the id");
  check(steer_config_id(a) != steer_config_id(cfg(true, 0.75f, 0x3eull, 9)), "the layer set is in the id");
  check(steer_config_id(a) != steer_config_id(cfg(true, 0.75f, 0x3full, 8)), "the vector identity is in the id");
  check(steer_config_id(a) != steer_config_id(cfg(false, 0.75f, 0x3full, 9)), "on/off is in the id");
  check(steer_config_id(cfg(false, 0.f, 0, 0)) != 0, "no configuration id is 0");
}

static void test_epochs() {
  SteerState s(cfg(true, 0.75f, 0x3full, 9));
  const uint64_t id0 = s.id();
  check(s.epoch() == 0, "a fresh state is epoch 0");
  check(s.check(0.75f) == SteerCheck::kNoop, "set-to-current is a no-op");
  check(s.epoch() == 0 && s.id() == id0, "a no-op moves no epoch and no id");
  check(s.check(1.0f) == SteerCheck::kApply, "1.0 applies");
  const uint64_t e1 = s.commit(1.0f);
  check(e1 == 1 && s.epoch() == 1, "an applied change moves the epoch to 1");
  check(s.config().alpha == 1.0f, "the configuration carries the new value");
  check(s.id() != id0, "the id changed with the configuration");
  // returning to the earlier value is a new change, never a revival of epoch 0
  const uint64_t e2 = s.commit(0.75f);
  check(s.id() != id0, "returning to old alpha cannot revive old cache identity");
  check(e2 == 2, "returning to 0.75 is epoch 2, not epoch 0");
  check(s.check(0.75f) == SteerCheck::kNoop && s.epoch() == 2, "and settable to itself again after that");
  // a second state does not see the first's epochs
  SteerState t(cfg(true, 0.75f, 0x3full, 9));
  check(t.epoch() == 0, "a new process starts at epoch 0");
}

static void test_off() {
  SteerState s(cfg(false, 0.f, 0, 0));
  check(s.check(0.75f) == SteerCheck::kSteeringOff, "steering off refuses a set");
  check(s.epoch() == 0, "a refused set moves no epoch");
}

// The actual production coordinator with fake rank/device transport, not a second implementation.
struct FourRanks {
  std::array<SteerState, 4> ranks;
  int refuse_apply = -1, refuse_recovery = -1, writes = 0, clears = 0;
  bool mismatch = false;
  FourRanks() { for (auto& s : ranks) s = SteerState(cfg(true, .75f, 63, 9)); }
  void transport(SteerOp op, float alpha, uint64_t expect, uint64_t target, SteerAck* all) {
    for (int r = 0; r < 4; ++r) {
      bool ok = true;
      if (op != SteerOp::kRead) {
        ok = op == SteerOp::kApply ? r != refuse_apply && ranks[r].epoch() == expect : r != refuse_recovery;
        if (ok) { ranks[r].recover(alpha, target); ++writes; }
      }
      all[r] = {};
      all[r].rank = r; all[r].ok = ok; all[r].alpha = ranks[r].config().alpha;
      all[r].epoch = ranks[r].epoch(); all[r].config_id = ranks[r].id();
      if (op == SteerOp::kRead && mismatch && r == 2) { all[r].alpha = 1.25f; all[r].ok = false; }
    }
  }
  SteerResult run(SteerTransaction& tx, const float* alpha) {
    return tx.run(ranks[0].config(), ranks[0].epoch(), alpha,
      [&](SteerOp op, float a, uint64_t e, uint64_t t, SteerAck* out) { transport(op, a, e, t, out); },
      [&] { ++clears; });
  }
};
static void test_multirank() {
  for (int failed = 0; failed < 4; ++failed) {
    FourRanks f;
    SteerTransaction tx;
    float alpha = 1.f;
    f.refuse_apply = failed;
    auto result = f.run(tx, &alpha);
    check(!result.ok && result.safe && !tx.blocked(), "partial transition is refused but recovered safely");
    check(f.clears == 1, "failed partial transition clears retained state");
    for (const auto& r : f.ranks)
      check(r.config().alpha == .75f && r.epoch() == 2, "applied and refusing ranks share recovery epoch 2");
  }
  FourRanks f;
  SteerTransaction tx;
  float alpha = 1.f;
  f.refuse_apply = 2; f.refuse_recovery = 3;
  auto result = f.run(tx, &alpha);
  check(!result.ok && !result.safe && tx.blocked() && f.clears == 1, "rollback failure clears state and fails closed");
  f.refuse_apply = -1; f.refuse_recovery = -1;
  result = f.run(tx, nullptr); // operator reconciliation, not an unrelated new transition
  check(result.ok && result.safe && !tx.blocked() && f.clears == 2, "explicit recovery confirms convergence before reopening");
  for (const auto& r : f.ranks) check(r.epoch() == 2 && r.config().alpha == .75f, "recovery target survives failed rollback");
  const int writes = f.writes, clears = f.clears;
  alpha = .75f;
  result = f.run(tx, &alpha);
  check(result.ok && result.safe && !result.changed && f.writes == writes && f.clears == clears,
        "same-alpha verified success neither writes nor evicts");
  alpha = 0.f;
  result = f.run(tx, &alpha);
  check(result.ok && result.changed && f.clears == clears + 1, "successful transition evicts exactly once");
  alpha = 2.01f;
  const int before = f.writes;
  result = f.run(tx, &alpha);
  check(!result.ok && result.safe && f.writes == before, "invalid value is refused before any rank write");
  FourRanks g; SteerTransaction t;
  g.mismatch = true;
  result = g.run(t, nullptr);
  check(!result.ok && result.safe && g.clears == 1, "GET detects device disagreement and recovers, rather than trusting rank zero");
  FourRanks different; SteerTransaction dt;
  different.ranks[2] = SteerState(cfg(true, .75f, 62, 9));
  result = different.run(dt, nullptr);
  check(!result.safe && dt.blocked(), "equal alpha/epoch cannot conceal a different layer/direction configuration");
  FourRanks exhausted; SteerTransaction et;
  for (auto& r : exhausted.ranks) r.recover(.75f, UINT64_MAX);
  exhausted.mismatch = true;
  result = exhausted.run(et, nullptr);
  check(!result.safe && et.blocked() && exhausted.writes == 0, "epoch exhaustion never wraps or reuses an identity during recovery");
}
static void test_device_recovery() {
  SteerState s(cfg(true, .75f, 63, 9));
  float host = .75f, device = .75f;
  bool confirmed = true;
  int graphs = 2, invalidations = 0, writes = 0;
  auto invalidate = [&] { graphs = 0; ++invalidations; };
  auto write = [&](float a) { check(graphs == 0 && !confirmed, "graphs invalidated before device write"); device = a; ++writes; };
  auto mismatch_read = [&] { return device + .25f; };
  check(!steer_device_apply(s, host, confirmed, 1.f, 1, invalidate, write, mismatch_read), "device mismatch is not success");
  check(!confirmed && host == 1.25f && s.config().alpha == host && s.epoch() == 1,
        "mismatch updates host/epoch to observed value and marks unconfirmed");
  // Even if the host looks like the desired rollback value, the device must be written again.
  host = .75f; s.recover(host, 1);
  check(steer_device_apply(s, host, confirmed, .75f, 2, invalidate, write, [&] { return device; }), "forced device recovery succeeds");
  check(device == .75f && host == .75f && confirmed && writes == 2 && invalidations == 2 && s.epoch() == 2,
        "host-equal recovery does not take a no-op shortcut");
}
static void test_store_enforcement() {
  using model::detail::StateRecords;
  StateRecords store;
  SteerTag original{cfg(true, .75f, 63, 9), 0};
  auto& ram = store.use_record(1, original);
  ram.saved = 128;
  auto& ssd = store.use_record(2, original);
  ssd.disk = true; ssd.dpages = {65536}; ssd.dsnaps.emplace(64, 131072); ssd.dend = 196608;
  check(store.find_current(1, original) && store.find_current(2, original), "RAM and SSD handles accepted with full identity");
  for (int field = 0; field < 5; ++field) {
    SteerTag changed = original;
    switch (field) {
      case 0: changed.config.on = false; break;
      case 1: changed.config.alpha = 1.f; break;
      case 2: changed.config.layers = 62; break;
      case 3: changed.config.vector_id = 10; break;
      default: changed.epoch = 2;
    }
    check(!store.find_current(1, changed) && !store.find_current(2, changed), "lookup refuses stale RAM and SSD identity");
    for (int h : {1, 2}) {
      bool refused = false;
      try { store.require_current(h, changed); } catch (const std::runtime_error&) { refused = true; }
      check(refused, "load/restore/unspill shared guard refuses stale record before access");
      refused = false;
      try { store.use_record(h, changed); } catch (const std::runtime_error&) { refused = true; }
      check(refused, "write/fork acquisition cannot relabel stale offsets or snapshots");
    }
    StateRecords::Kept kept; kept.steering = original; kept.pages = {{65536, 17}};
    check(!kept.steering.matches(changed), "kept-record identity includes every configuration field and epoch");
  }
  check(store.require_current(2, original).dpages[0] == 65536, "refusal preserves SSD ownership, never reassigns it");
  check(steer_config_id(original.config, 0) != steer_config_id(original.config, 2), "A-B-A epoch is part of configuration hash");
}

// Only the command envelope, broadcast and storage effects are stand-ins. All clear -> batch ->
// issue -> apply -> spill decisions below are the SAME templates used by serve.cpp and cmd.cpp.
struct CpuCommand {
  int32_t op = 0, slot = -1, handle = -1;
  int32_t live[model::kMaxSlots] = {};
  int32_t n_ops = 0;
  srv::StateOp ops[srv::kMaxOps] = {};
};
struct CpuStore {
  model::detail::StateRecords records;
  SteerState* state = nullptr;
  std::vector<int> pinned;
  int frees = 0, spills = 0, selected = 0;
  bool disk_enabled = true;
  size_t state_disk_capacity() const { return disk_enabled ? 1 : 0; }
  void use_slot(int) { ++selected; }
  void state_pin(const std::vector<int>& h) { pinned = h; }
  std::vector<int> state_sessions() const {
    std::vector<int> out;
    for (const auto& [h, _] : records.sess) out.push_back(h);
    return out;
  }
  void state_free(int h) { frees += records.sess.erase(h); }
  void state_spill(int h) {
    ++spills;
    auto it = records.sess.find(h);
    if (it == records.sess.end() || it->second.disk) return;
    // Same early exit and actual production tag guard as state.cu, before any physical SSD I/O.
    records.require_current(h, {state->config(), state->epoch()}).disk = true;
  }
};
struct RetainedRanks : FourRanks {
  struct Session { int h; };
  std::array<CpuStore, 4> stores;
  std::map<uint64_t, Session> sessions;
  std::vector<uint64_t> slots = std::vector<uint64_t>(model::kMaxSlots, 0);
  std::set<uint64_t> later;
  int batches = 0, later_live_batches = 0, recovery_after_clear = 0;
  std::array<int, 4> applied_batches{};
  explicit RetainedRanks(int handles) {
    for (int r = 0; r < 4; ++r) {
      auto& m = stores[r]; m.state = &ranks[r];
      for (int h = 1; h <= handles; ++h) {
        auto& record = m.records.use_record(h, {ranks[r].config(), ranks[r].epoch()});
        // The review's occupancy: the first 12 may already be on SSD; handle 13 is still RAM.
        record.saved = 128;
        record.disk = h <= srv::kMaxOps;
        if (record.disk) { record.dpages = {size_t(h) * 65536}; record.kept = true; }
        else record.snaps.emplace(64, nullptr); // RAM snapshot metadata; no CUDA allocation on CPU
      }
      m.pinned = {13};
    }
    for (int h = 1; h <= handles; ++h) sessions.emplace(h, Session{h});
    for (int h = 1; h <= std::min(handles, model::kMaxSlots); ++h) slots[h-1] = h;
    later.insert(13);
  }
  void apply(int rank, const CpuCommand& c) {
    srv::apply_state_command(stores[rank], c, [](CpuStore& m, const srv::StateOp& op) {
      if (op.op != srv::kStFree) throw std::runtime_error("unexpected test operation");
      m.state_free(op.h);
    });
    ++applied_batches[rank];
  }
  void issue(CpuCommand& c) {
    srv::issue_state_command(c, sessions, slots, [&](const CpuCommand& cmd) {
      ++batches;
      if (cmd.live[12] == 13) ++later_live_batches;
      // Followers enter the same apply path, before dispatch, just as follow() does.
      for (int r = 1; r < 4; ++r) apply(r, cmd);
    }, [&](const CpuCommand& cmd) { apply(0, cmd); });
  }
  void clear() {
    ++clears;
    srv::clear_retained_state<CpuCommand>(stores[0], sessions, slots, later,
                                         [&](CpuCommand& c) { issue(c); });
  }
  SteerResult run_retained(SteerTransaction& tx, const float* alpha) {
    return tx.run(ranks[0].config(), ranks[0].epoch(), alpha,
      [&](SteerOp op, float a, uint64_t e, uint64_t t, SteerAck* out) {
        if (op == SteerOp::kRecover) {
          for (const auto& m : stores) check(m.records.sess.empty(), "all ranks freed before recovery writes");
          check(sessions.empty() && later.empty(), "metadata cleared before recovery writes");
          ++recovery_after_clear;
        }
        transport(op, a, e, t, out);
      }, [&] { clear(); });
  }
};
static void test_epoch_teardown_pipeline() {
  // Control: the unchanged ordinary path really hits the reported stale RAM spill, before frees.
  RetainedRanks old(13);
  for (auto& r : old.ranks) r.recover(1.f, 1);
  std::vector<srv::StateOp> frees;
  for (const int h : old.stores[0].state_sessions()) frees.push_back({srv::kStFree, h, 0, 0});
  bool stale_spill = false;
  try {
    srv::state_command_batches<CpuCommand>(frees, srv::kCmdState, [&](CpuCommand& c) { old.issue(c); });
  } catch (const std::runtime_error&) { stale_spill = true; }
  check(stale_spill && old.stores[1].spills == 1, "ordinary clear reproduces stale handle-13 spill");
  for (const auto& m : old.stores) check(m.frees == 0 && m.records.sess.size() == 13,
                                        "control fails before first free on every rank");
  check(old.sessions.size() == 13 && old.slots[12] == 13, "control retains old live metadata");

  // Success and a refusal on EACH rank, including rank zero; also cover a third/tail batch.
  for (const int handles : {13, 16, 25}) for (int failed = -1; failed < 4; ++failed) {
    RetainedRanks f(handles);
    f.refuse_apply = failed;
    SteerTransaction tx;
    float alpha = 1.f;
    auto result = f.run_retained(tx, &alpha);
    check(result.ok == (failed == -1) && result.safe && !tx.blocked(), "clear permits full apply or safe rollback");
    check(f.clears == 1 && f.batches == (handles + srv::kMaxOps - 1) / srv::kMaxOps,
          "production teardown uses all full and tail batches");
    check(f.later_live_batches == f.batches, "handle 13 remains slot-live through teardown batches");
    for (int r = 0; r < 4; ++r) {
      const auto& m = f.stores[r];
      check(m.frees == handles && m.records.sess.empty() && m.pinned.empty(), "all ranks free all old RAM/SSD/kept records");
      check(m.spills == 0 && m.selected == 0 && f.applied_batches[r] == f.batches,
            "teardown never selects slots, spills or acquires stale snapshots on any rank");
      check(f.ranks[r].epoch() == (failed == -1 ? 1u : 2u) &&
            f.ranks[r].config().alpha == (failed == -1 ? 1.f : .75f), "common applied or recovery identity");
    }
    check(f.sessions.empty() && f.later.empty() && std::all_of(f.slots.begin(), f.slots.end(), [](auto h) { return !h; }),
          "server session, slot and deferred metadata cleared");
    check(f.recovery_after_clear == (failed == -1 ? 0 : 1), "partial apply reaches recovery after actual clear");
  }

  RetainedRanks retry(25);
  retry.refuse_apply = 2; retry.refuse_recovery = 3;
  SteerTransaction tx;
  float alpha = 1.f;
  auto result = retry.run_retained(tx, &alpha);
  check(!result.ok && !result.safe && tx.blocked(), "failed recovery remains closed after real teardown");
  for (const auto& m : retry.stores) check(m.records.sess.empty() && m.frees == 25, "failed recovery still frees every rank");
  retry.refuse_recovery = -1;
  result = retry.run_retained(tx, nullptr);
  check(result.ok && result.safe && !tx.blocked() && retry.clears == 2 && retry.batches == 3,
        "empty teardown retry permits common-epoch reconciliation without stale acquisitions");

  RetainedRanks noop(13); SteerTransaction nt;
  alpha = .75f;
  result = noop.run_retained(nt, &alpha);
  check(result.ok && !result.changed && noop.batches == 0 && noop.sessions.size() == 13,
        "confirmed same-alpha exits before clear/batching/eviction");
  for (const auto& m : noop.stores) check(m.records.sess.size() == 13 && m.frees == 0 && m.spills == 0,
                                         "same-alpha preserves every rank's records");

  // Guard the bypass itself: it accepts frees only, and validates ALL ops before any free.
  RetainedRanks invalid(13);
  CpuCommand c; c.op = srv::kCmdEpochFree; c.n_ops = 2;
  c.ops[0] = {srv::kStFree, 1}; c.ops[1] = {srv::kStSpill, 13};
  bool rejected = false;
  try { invalid.apply(0, c); } catch (const std::runtime_error&) { rejected = true; }
  check(rejected && invalid.stores[0].frees == 0 && invalid.stores[0].spills == 0,
        "teardown cannot smuggle spill/retain/acquisition past stale-tag guards");
  std::printf("epoch teardown pipeline: ordinary stale-spill control, 15 full/partial cases, retry, no-op, free-only guard: ok\n");
}

int main() {
  test_parse();
  test_check();
  test_identity();
  test_epochs();
  test_off();
  test_multirank();
  test_device_recovery();
  test_store_enforcement();
  test_epoch_teardown_pipeline();
  if (g_fail) {
    std::fprintf(stderr, "steer_test: %d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("steer_test: ok\n");
  return 0;
}
