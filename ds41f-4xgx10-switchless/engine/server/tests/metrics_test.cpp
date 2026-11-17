// CPU-only contract test. This does NOT qualify GPU serving or throughput.
#include "server/metrics.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
using namespace pico::metrics;
uint64_t count(const Histogram& h) {
  uint64_t n=0; for(const auto& b:h.bins) n+=read(b); return n;
}
int main(int argc, char**) {
  if(argc>1) {
    Endpoint ep("127.0.0.1:0","telemetry-test",0);
    set(registry.initialization,1); set(registry.slots_capacity,16);
    std::cout<<ep.port()<<std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::vector<RequestMetrics> requests(19);
    for(auto& r:requests) r.receive(steady_us());
    for(int i=0;i<16;++i) {requests[i].start(steady_us());requests[i].token(steady_us());requests[i].token(steady_us());}
    set(registry.slots,16); progress(Decode);
    // Deliberately block the owner for >120s: listener must NOT need it to answer.
    std::this_thread::sleep_for(std::chrono::seconds(125));
    for(int i=0;i<19;++i) requests[i].finish(i==0?4:i<16?0:3,steady_us());
    set(registry.slots,0);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    return 0;
  }
  Histogram h;
  for(uint64_t us:{0ull,1000ull,1001ull,600000000ull,600000001ull}) h.observe(us);
  assert(count(h)==5 && read(h.bins[0])==2 && read(h.bins[1])==1 && read(h.bins[19])==1);
  const uint64_t first=1000000;
  RequestMetrics a; a.receive(first); assert(read(registry.waiting)==1);
  a.start(first+1000); a.start(first+2000); // resume must not become a second request
  assert(read(registry.running)==1 && read(registry.waiting)==0 && count(registry.queue)==1);
  a.token(first+4000); a.token(first+10000); a.token(first+20000);
  assert(read(registry.generated)==3 && count(registry.ttft)==1);
  a.finish(0,first+30000); a.finish(0,first+40000); a.token(first+40001);
  assert(read(registry.running)==0 && read(registry.generated)==3 && read(registry.outcomes[0])==1);
  assert(count(registry.tpot)==1 && read(registry.tpot.sum_us)==13000);
  // Pre-admission failures and cancellation: no TTFT / TPOT sample and no unsigned gauge underflow.
  for(size_t why=1;why<6;++why) {RequestMetrics b;b.receive(first);b.finish(why,first+1);}
  assert(read(registry.waiting)==0 && count(registry.ttft)==1 && count(registry.tpot)==1);
  // Histogram bins cannot lose updates with multiple independent producers.
  Histogram concurrent;
  std::vector<std::thread> writers;
  for(int i=0;i<16;++i) writers.emplace_back([&]{for(int k=0;k<10000;++k) concurrent.observe(k);});
  for(auto& t:writers) t.join();
  assert(count(concurrent)==160000);
  Endpoint ep("", "a\"b\\c\nd", 0);
  const auto text=ep.render();
  assert(text.find("model_name=\"a\\\"b\\\\c\\nd\"")!=std::string::npos);
  assert(text.find("pico:kv_cache_usage_perc{")==std::string::npos);
  assert(text.size()<(256<<10));
  Endpoint follower("", "test", 1);
  assert(follower.render().find("pico:generation_tokens_total{")==std::string::npos);
  std::cout<<"PASS histogram boundaries, 16-writer accounting, exactly-once lifecycle, follower exclusion, labels, response budget\n";
}
