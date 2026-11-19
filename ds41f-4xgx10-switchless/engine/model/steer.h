// GLP steering, runtime control (docs/steering-control.md).
//
// The forward pass reads one effective configuration: whether steering is on, one scalar alpha, the
// steered layers and the direction file's identity (glp.cpp). Startup reads it from
// WEIGHTLESS_STEER_PATH / WEIGHTLESS_STEER_ALPHA / WEIGHTLESS_STEER_LAYERS. This module turns that
// constant into an operator-settable value that a running process can change once, globally, for all
// traffic, at a step boundary - without a restart and without touching serving numerics other than the
// requested alpha.
//
// Two identities, both CPU-only and unit-tested (tests/steer_test.cpp):
//   * the configuration id: FNV-1a over (on, alpha bits, layer mask, direction identity, epoch). Cached KV is
//     only valid for the configuration it was computed under, so the id tags retained state.
//   * the epoch: a monotonic counter bumped by every applied change. State from an earlier epoch is
//     never served, continued, forked or restored, even if the configuration's numbers return to an
//     earlier value. Each applied set starts a new epoch.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace model {

// The accepted range of a runtime alpha, inclusive. GLP v1 is projective removal h - a (h.d) d with a
// unit direction: 1.0 removes the whole component, values above it over-remove (the component flips
// sign) and 0.0 is the identity. 2.0 is the documented ceiling: past it the projection is a sign flip of
// the removed component rather than a strength, and nothing measured exists beyond 1.0. The startup
// path keeps its older, wider contract (a finite float); this range judges runtime set requests only.
inline constexpr std::array<float, 2> kSteerAlphaRange = {0.0f, 2.0f};

// A set request's epoch expectation: the value must match the engine's epoch, unless it is this, which
// only the rollback after a failed multi-rank set passes (the ranks may have diverged; the value is
// restored, not compared).
inline constexpr uint64_t kSteerAnyEpoch = ~0ull;

// The effective configuration a forward pass runs under.
struct SteerConfig {
  bool on = false;
  float alpha = 0.f;
  uint64_t layers = 0;     // bit L set: layer L is steered
  uint64_t vector_id = 0;  // the direction file's identity (its content), 0 when steering is off
};

bool operator==(const SteerConfig& a, const SteerConfig& b);
inline bool operator!=(const SteerConfig& a, const SteerConfig& b) { return !(a == b); }

// FNV-1a over the configuration's canonical bytes. Equal ids mean equal behaviour: the same steering
// numeric path, the same layers, the same directions and the same epoch.
uint64_t steer_config_id(const SteerConfig& c, uint64_t epoch = 0);

// The direction file's identity: FNV-1a over its validated unit directions. 0 for no directions (off).
uint64_t steer_directions_id(const float* dir, size_t n);

// A strict alpha parse: the whole string, no trailing text, finite, in [kSteerAlphaRange[0],
// kSteerAlphaRange[1]]. `why` names the reason on refusal.
bool steer_parse_alpha(const std::string& text, float* out, std::string* why);

// What a set request would do, decided before any rank is touched.
enum class SteerCheck {
  kApply,       // a different alpha, steering on, in range: apply it
  kNoop,        // the same alpha: nothing changes, no epoch, no state dropped
  kSteeringOff, // steering is off; alpha alone never enables it
  kBadAlpha,    // not finite, or outside [kSteerAlphaRange[0], kSteerAlphaRange[1]]
};

SteerCheck steer_check(const SteerConfig& c, float alpha);

// The engine's steering state machine. One instance per process (Model::Impl::steer_).
class SteerState {
 public:
  SteerState() = default;
  explicit SteerState(const SteerConfig& c) : cfg_(c), id_(steer_config_id(c)) {}

  const SteerConfig& config() const { return cfg_; }
  uint64_t epoch() const { return epoch_; }
  uint64_t id() const { return id_; }

  SteerCheck check(float alpha) const { return steer_check(cfg_, alpha); }

  // Commits a change that `check` returned kApply for: the epoch moves and the id is recomputed.
  // Returns the new epoch.
  uint64_t commit(float alpha);
  void recover(float alpha, uint64_t epoch);

 private:
  SteerConfig cfg_{};
  uint64_t epoch_ = 0;
  uint64_t id_ = 0;
};

// Shared device-write transaction, CPU-failure-injected in steer_test. Callers serialize/sync
// first. Invalidating before writing protects graph replay even if read-back fails.
bool steer_device_apply(SteerState& state, float& host_alpha, bool& confirmed, float alpha,
                        uint64_t target_epoch, const std::function<void()>& invalidate,
                        const std::function<void(float)>& write, const std::function<float()>& read);

// A complete cache tag, shared by live RAM records, SSD offsets and kept records.
struct SteerTag {
  SteerConfig config{};
  uint64_t epoch = 0;
  bool matches(const SteerTag& other) const { return config == other.config && epoch == other.epoch; }
};

// CPU-testable production coordinator. Transport runs the same operation on all four ranks.
struct SteerAck {
  float alpha = 0.f;
  uint64_t epoch = 0, config_id = 0;
  int32_t ok = 0, rank = 0;
  char why[96] = {};
};
bool steer_confirmed(const SteerAck* a, const SteerConfig& c, uint64_t epoch);
enum class SteerOp { kRead, kApply, kRecover };
using SteerTransport = std::function<void(SteerOp, float, uint64_t, uint64_t, SteerAck*)>;
struct SteerResult {
  bool ok = false, safe = false, changed = false;
  std::string why;
  SteerAck ranks[4]{};
};
class SteerTransaction {
 public:
  SteerResult run(SteerConfig current, uint64_t epoch, const float* alpha,
                  const SteerTransport& transport, const std::function<void()>& clear);
  bool blocked() const { return pending_; }
 private:
  bool pending_ = false;
  SteerConfig recovery_{};
  uint64_t recovery_epoch_ = 0;
};

}  // namespace model
