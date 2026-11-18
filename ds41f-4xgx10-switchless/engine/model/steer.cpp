// GLP runtime steering control: identity, validation and the epoch machine (steer.h).
#include "model/steer.h"

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

uint64_t steer_config_id(const SteerConfig& c) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64 offset basis
  const uint8_t on = c.on ? 1 : 0;
  h = fnv1a(h, &on, sizeof on);
  // The alpha's bits, not its value: 0.0 and -0.0 are different device buffers, so they are not equal.
  h = fnv1a(h, &c.alpha, sizeof c.alpha);
  h = fnv1a(h, &c.layers, sizeof c.layers);
  h = fnv1a(h, &c.vector_id, sizeof c.vector_id);
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
  if (v < kSteerAlphaMin || v > kSteerAlphaMax) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "alpha %g is outside the accepted range [%g, %g]", (double)v,
                  (double)kSteerAlphaMin, (double)kSteerAlphaMax);
    *why = buf;
    return false;
  }
  *out = v;
  return true;
}

SteerCheck steer_check(const SteerConfig& c, float alpha) {
  if (!std::isfinite(alpha) || alpha < kSteerAlphaMin || alpha > kSteerAlphaMax) return SteerCheck::kBadAlpha;
  if (!c.on) return SteerCheck::kSteeringOff;
  if (std::memcmp(&alpha, &c.alpha, sizeof alpha) == 0) return SteerCheck::kNoop;  // -0.0 is not 0.0
  return SteerCheck::kApply;
}

uint64_t SteerState::commit(float alpha) {
  cfg_.alpha = alpha;
  ++epoch_;
  id_ = steer_config_id(cfg_);
  return epoch_;
}

}  // namespace model
