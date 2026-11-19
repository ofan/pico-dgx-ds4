// GLP runtime steering control: identity, validation and the epoch machine (steer.h).
#include "model/steer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace model {
namespace {

uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

// A finite float, and no trailing characters after the number.
bool parse_strict(std::string_view text, float* out) {
  if (text.empty()) return false;
  if (std::isspace(static_cast<unsigned char>(text.front()))) return false;  // strtof would skip it
  std::string tmp(text);
  char* end = nullptr;
  const float v = std::strtof(tmp.c_str(), &end);
  if (end != tmp.c_str() + tmp.size()) return false;  // "", "1.0x", "1.0 ", "nanx"
  if (!std::isfinite(v)) return false;
  *out = v;
  return true;
}

}  // namespace

bool operator==(const SteerConfig& a, const SteerConfig& b) {
  return a.on == b.on && std::memcmp(&a.alpha, &b.alpha, sizeof a.alpha) == 0 && a.layers == b.layers &&
         a.vector_id == b.vector_id;
}

uint64_t steer_config_id(const SteerConfig& c, uint64_t epoch) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64 offset basis
  const uint8_t on = c.on ? 1 : 0;
  h = fnv1a(h, &on, sizeof on);
  // The alpha's bits, not its value: 0.0 and -0.0 are different device buffers, so they are not equal.
  h = fnv1a(h, &c.alpha, sizeof c.alpha);
  h = fnv1a(h, &c.layers, sizeof c.layers);
  h = fnv1a(h, &c.vector_id, sizeof c.vector_id);
  h = fnv1a(h, &epoch, sizeof epoch);
  return h ? h : 1;  // 0 is reserved for "no identity"
}

uint64_t steer_directions_id(const float* dir, size_t n) {
  if (!dir || n == 0) return 0;
  const uint64_t h = fnv1a(1469598103934665603ull, dir, n * sizeof(float));
  return h ? h : 1;
}

bool steer_parse_alpha(const std::string& text, float* out, std::string* why) {
  float v = 0.f;
  if (!parse_strict(text, &v)) {
    *why = "alpha is not a finite number: '" + text + "'";
    return false;
  }
  if (v < kSteerAlphaRange[0] || v > kSteerAlphaRange[1]) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "alpha %g is outside the accepted range [%g, %g]", (double)v,
                  (double)kSteerAlphaRange[0], (double)kSteerAlphaRange[1]);
    *why = buf;
    return false;
  }
  *out = v;
  return true;
}

SteerCheck steer_check(const SteerConfig& c, float alpha) {
  if (!std::isfinite(alpha) || alpha < kSteerAlphaRange[0] || alpha > kSteerAlphaRange[1]) return SteerCheck::kBadAlpha;
  if (!c.on) return SteerCheck::kSteeringOff;
  if (std::memcmp(&alpha, &c.alpha, sizeof alpha) == 0) return SteerCheck::kNoop;  // -0.0 is not 0.0
  return SteerCheck::kApply;
}

uint64_t SteerState::commit(float alpha) {
  cfg_.alpha = alpha;
  ++epoch_;
  id_ = steer_config_id(cfg_, epoch_);
  return epoch_;
}

void SteerState::recover(float alpha, uint64_t epoch) {
  cfg_.alpha = alpha;
  epoch_ = epoch;
  id_ = steer_config_id(cfg_, epoch_);
}

bool steer_device_apply(SteerState& state, float& host_alpha, bool& confirmed, float alpha,
                        uint64_t target_epoch, const std::function<void()>& invalidate,
                        const std::function<void(float)>& write, const std::function<float()>& read) {
  confirmed = false;
  invalidate();
  write(alpha);  // deliberately unconditional: recovery must repair a host-equal/device-different state
  host_alpha = read();
  state.recover(host_alpha, target_epoch);
  confirmed = std::memcmp(&host_alpha, &alpha, sizeof(float)) == 0;
  return confirmed;
}

bool steer_confirmed(const SteerAck* a, const SteerConfig& c, uint64_t epoch) {
  for (int r = 0; r < 4; ++r)
    if (!a[r].ok || a[r].rank != r || a[r].epoch != epoch ||
        std::memcmp(&a[r].alpha, &c.alpha, sizeof(float)) || a[r].config_id != steer_config_id(c, epoch)) return false;
  return true;
}

SteerResult SteerTransaction::run(SteerConfig current, uint64_t epoch, const float* alpha,
                                 const SteerTransport& transport, const std::function<void()>& clear) {
  SteerResult out;
  auto recover = [&] {
    // Retained state is cleared even when recovery itself fails. No old RAM or SSD record survives.
    clear();
    if (!recovery_epoch_) { out.why += "; epoch exhausted; admission closed"; return; }
    transport(SteerOp::kRecover, recovery_.alpha, kSteerAnyEpoch, recovery_epoch_, out.ranks);
    out.safe = steer_confirmed(out.ranks, recovery_, recovery_epoch_);
    pending_ = !out.safe;
    out.why += out.safe ? "; recovered all four ranks at a common epoch" : "; recovery unconfirmed; admission closed";
  };
  if (pending_) {
    out.why = "retrying the pending recovery (new sets require a subsequent command)";
    recover();
    out.ok = !alpha && out.safe;
    return out;
  }
  transport(SteerOp::kRead, 0.f, 0, 0, out.ranks);
  const bool initial_ok = steer_confirmed(out.ranks, current, epoch);
  if (initial_ok && !alpha) { out.ok = out.safe = true; return out; }
  if (initial_ok && alpha) {
    const auto check = steer_check(current, *alpha);
    if (check == SteerCheck::kBadAlpha || check == SteerCheck::kSteeringOff) {
      out.safe = true;
      out.why = check == SteerCheck::kBadAlpha ? "alpha outside the finite runtime range [0, 2]" : "steering is off";
      return out;
    }
    if (check == SteerCheck::kNoop) { out.ok = out.safe = true; return out; }
    if (epoch >= UINT64_MAX - 2) { out.safe = true; out.why = "steering epoch exhausted"; return out; }
    SteerConfig target = current;
    target.alpha = *alpha;
    transport(SteerOp::kApply, *alpha, epoch, epoch + 1, out.ranks);
    out.changed = true;
    if (steer_confirmed(out.ranks, target, epoch + 1)) {
      clear();
      out.ok = out.safe = true;
      return out;
    }
  }
  // Even an apparent host no-op may hide a written-but-unconfirmed device scalar.
  out.why = initial_ok ? "partial steering transition refused" : "initial device/configuration agreement failed";
  uint64_t highest = epoch;
  for (const auto& a : out.ranks) highest = std::max(highest, a.epoch);
  recovery_ = current;
  recovery_epoch_ = highest == UINT64_MAX ? 0 : highest + 1;
  pending_ = true;
  recover();
  return out;
}

}  // namespace model
