#!/usr/bin/env python3
"""CPU control-flow regression for the actual cmd.cpp teacher-forcing function.

Extracts, without rewriting, batch_tf_loop and its MetricsSlots RAII guard.
Only the model/CUDA boundary is a double. This does NOT prove GPU token parity.
Use --out for a new scratch directory; compiler inputs and binary are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
cmd = (root / 'engine/server/cmd.cpp').read_text()
start = cmd.index('struct MetricsSlots {')
guard = cmd[start:cmd.index('\n};', start) + 3]
start = cmd.index('std::vector<float> batch_tf_loop(')
body = cmd[start:cmd.index('// B3 (cmd.h)', start)].strip()
a.out.mkdir(parents=True, exist_ok=False)
prelude = r'''
#include "server/metrics.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace pico::metrics;
namespace model {
namespace ops { constexpr int kMaxSess=16; }
constexpr int kMaxSessRows=16, kMaxRows=256;
struct SampleOut {};
struct Sink {};
struct Batch {
  int R=0, nsess=0, sp[16]{}, tokens[256]{}, pos[256]{}, slot[256]{}, sess[256]{};
  unsigned flags=0;
  Sink* sink=nullptr;
};
struct Model {
  std::vector<int> active, rows;
  SampleOut out;
  int launch_batch(const Batch& b) {
    assert(read(registry.slots)==unsigned(b.nsess));
    Endpoint ep("", "teacher-forcing-test", 0);
    const auto text=ep.render();
    assert(text.find("pico:slots_active{model_name=\"teacher-forcing-test\",rank=\"0\"} "+
                     std::to_string(b.nsess)+"\n")!=std::string::npos);
    assert(text.find("# HELP pico:slots_active Entries participating in decode (including teacher forcing and draining rows); zero outside decode.\n")!=std::string::npos);
    active.push_back(b.nsess); rows.push_back(b.R);
    return 0;
  }
  const SampleOut& wait(int) {return out;}
  float last_gpu_ms() {return 1.0f;}
};
}
struct Cmd {int rows=2; unsigned flags=0;};
struct BatchEntry {int max_new=0, pos0=0, limit=10, sp=0, slot=0;};
struct Watchdog {double started=0;};
using OnBatch=std::function<void(int,const model::SampleOut&,int)>;
double now_s() {return 1;}
template<class... T> [[noreturn]] void die(const char*,T...) {throw std::runtime_error("invalid batch");}
'''
checks = r'''
int main() {
  const std::vector<BatchEntry> e={{5,0,10,0,0},{3,0,10,0,1},{1,0,10,0,2},{0,0,10,0,3}};
  const std::vector<std::vector<int32_t>> tokens={{1,2,3,4,5},{6,7,8},{9},{}};
  model::Model m; Watchdog wd; std::vector<int> callbacks(4);
  const auto ms=batch_tf_loop(m,Cmd{},e,tokens,wd,[&](int k,const model::SampleOut&,int){
    assert(read(registry.slots)==unsigned(m.active.back())); ++callbacks[k];
  },nullptr);
  assert((m.active==std::vector<int>{3,2,1}));
  assert((m.rows==std::vector<int>{5,3,1})); // entries != token rows
  assert((callbacks==std::vector<int>{5,3,1,0}));
  assert(ms.size()==3 && wd.started==0 && read(registry.slots)==0);
  std::cout<<"PASS teacher-forcing slots: 0 -> 3 -> 2 -> 1 -> 0; rows: 5,3,1; rendered HELP matches contract\n";
  model::Model empty;
  batch_tf_loop(empty,Cmd{},{{0,0,10,0,0}},{{}},wd,{},nullptr);
  assert(empty.active.empty() && read(registry.slots)==0);
  model::Model unwind; bool threw=false;
  try {
    batch_tf_loop(unwind,Cmd{},e,tokens,wd,[](int,const model::SampleOut&,int){
      throw std::runtime_error("test callback exit");
    },nullptr);
  } catch(const std::runtime_error&) {threw=true;}
  assert(threw && read(registry.slots)==0);
  std::cout<<"PASS zero-length entries, zero outside decode, unwind reset; no GPU execution\n";
}
'''
source=a.out / 'teacher_forcing.cpp'
source.write_text(prelude + guard + '\n' + body + '\n' + checks)
print(json.dumps({'extracted_batch_tf_sha256': hashlib.sha256(body.encode()).hexdigest(),
                  'test_kind': 'actual scheduler body; CPU model double; NOT GPU parity'}), flush=True)
command=[os.environ.get('CXX','g++'), '-std=c++20', '-O2','-Wall','-Wextra','-Werror',
         '-pthread','-I',str(root/'engine'),str(source),str(root/'engine/server/metrics.cpp'),
         '-o',str(a.out/'teacher_forcing_test')]
print('Compile:', ' '.join(command), flush=True)
subprocess.run(command, check=True)
subprocess.run([str(a.out/'teacher_forcing_test')], check=True)
