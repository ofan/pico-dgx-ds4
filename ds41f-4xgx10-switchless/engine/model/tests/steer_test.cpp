// Host-only tests of the runtime steering control (model/steer.h): the strict alpha parser, the
// documented range, the configuration identity, and the epoch machine. No GPU, no CUDA, no model.
//
// What this proves: a set request is accepted or refused exactly as documented, an identical value is
// a no-op that moves no epoch, every applied change moves the epoch once, and the configuration id
// changes if and only if the effective configuration changes. What it cannot prove: that the engine
// reads the value back off the device, that CUDA graphs are invalidated, or anything about throughput.
#include "model/steer.h"

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

int main() {
  test_parse();
  test_check();
  test_identity();
  test_epochs();
  test_off();
  if (g_fail) {
    std::fprintf(stderr, "steer_test: %d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("steer_test: ok\n");
  return 0;
}
