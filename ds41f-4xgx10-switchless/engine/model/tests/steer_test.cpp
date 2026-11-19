// CPU checks for production steering validation, hashing, four-rank coordination, device transaction
// and cache-record enforcement. Device/transport stand-ins are not GPU or physical SSD qualification.
#include "model/steer.h"
#include "model/state_records.h"
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

int main() {
  test_parse();
  test_check();
  test_identity();
  test_epochs();
  test_off();
  test_multirank();
  test_device_recovery();
  test_store_enforcement();
  if (g_fail) {
    std::fprintf(stderr, "steer_test: %d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("steer_test: ok\n");
  return 0;
}
