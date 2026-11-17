#include "server/metrics.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <iomanip>
#include <future>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pico::metrics {
namespace {
std::string escape(const std::string& s) {
  if(s.size()>256) throw std::invalid_argument("metrics model name exceeds 256 bytes");
  std::string out;
  for(char c:s) { if(c=='\\'||c=='"') out+='\\'; if(c=='\n') out+="\\n"; else out+=c; }
  return out;
}
struct Client {
  int fd=-1;
  uint64_t deadline=0;
  std::string input, output;
  size_t sent=0;
};
}
Endpoint::Endpoint(std::string address,std::string model_name,int rank,int cpu)
    : labels_("model_name=\""+escape(model_name)+"\",rank=\""+std::to_string(rank)+"\""),
      started_(wall_us()), cpu_(cpu), rank_(rank) {
  if(address.empty()) return; // explicitly disabled; numeric hooks still have bounded cost
  const auto colon=address.rfind(':');
  if(colon==std::string::npos) throw std::invalid_argument("metrics address must be IPv4:port");
  const std::string host=address.substr(0,colon), ps=address.substr(colon+1);
  size_t used=0; const int port=std::stoi(ps,&used);
  if(used!=ps.size()||port<0||port>65535) throw std::invalid_argument("bad metrics port");
  sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_port=htons(port);
  if(inet_pton(AF_INET,host.c_str(),&addr.sin_addr)!=1||addr.sin_addr.s_addr==INADDR_ANY)
    throw std::invalid_argument("metrics needs an explicit non-wildcard IPv4 address");
  cpu_set_t available; CPU_ZERO(&available);
  if(pthread_getaffinity_np(pthread_self(),sizeof available,&available)!=0)
    throw std::runtime_error("cannot read metrics CPU affinity");
  auto reserved=[](int i) { return (i>=5&&i<=9)||i==12||i==15; };
  if(cpu_<0) { // captured before main pins inference; never silently inherit CPU 15
    for(int i=0;i<CPU_SETSIZE;++i) if(!reserved(i)&&CPU_ISSET(i,&available)) { cpu_=i; break; }
  }
  if(cpu_<0||cpu_>=CPU_SETSIZE||!CPU_ISSET(cpu_,&available)||reserved(cpu_))
    throw std::invalid_argument("metrics CPU must be allowed and distinct from reserved 5-9 / 12 / 15");
  fd_=socket(AF_INET,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
  if(fd_<0) throw std::runtime_error("cannot create metrics socket");
  int yes=1; setsockopt(fd_,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof yes);
  if(bind(fd_,reinterpret_cast<sockaddr*>(&addr),sizeof addr)!=0||listen(fd_,4)!=0) {
    close(fd_); fd_=-1; throw std::runtime_error("cannot bind metrics endpoint");
  }
  socklen_t n=sizeof addr; getsockname(fd_,reinterpret_cast<sockaddr*>(&addr),&n); port_=ntohs(addr.sin_port);
  std::fprintf(stderr,"native metrics: port %d CPU %d\n",port_,cpu_);
  std::promise<void> ready;
  auto initialized=ready.get_future();
  try {
    thread_=std::thread([this, ready=std::move(ready)]() mutable {
      cpu_set_t cpus; CPU_ZERO(&cpus); CPU_SET(cpu_,&cpus);
      if(pthread_setaffinity_np(pthread_self(),sizeof cpus,&cpus)!=0) {
        ready.set_exception(std::make_exception_ptr(std::runtime_error("cannot set metrics affinity")));
        return;
      }
      ready.set_value();
      try { run(); }
      catch(const std::exception& e) { std::fprintf(stderr,"native metrics stopped: %s\n",e.what()); }
    });
    initialized.get();
  } catch(...) {
    stop_.store(true,std::memory_order_relaxed);
    if(thread_.joinable()) thread_.join();
    close(fd_); fd_=-1; throw;
  }
}
Endpoint::~Endpoint() {
  stop_.store(true,std::memory_order_relaxed);
  if(thread_.joinable()) thread_.join();
  if(fd_>=0) close(fd_);
}
std::string Endpoint::render() const {
  std::ostringstream o; o<<std::setprecision(15);
  auto scalar=[&](const char* n,const char* type,const char* help,double value,const std::string& extra="") {
    o<<"# HELP pico:"<<n<<' '<<help<<"\n# TYPE pico:"<<n<<' '<<type<<"\npico:"<<n<<'{'<<labels_<<extra<<"} "<<value<<'\n';
  };
  auto gauge=[&](const char* n,const char* help,const Atomic& a) { scalar(n,"gauge",help,read(a)); };
  auto counter=[&](const char* n,const char* help,const Atomic& a) { scalar(n,"counter",help,read(a)); };
  scalar("metrics_heartbeat_timestamp_seconds","gauge","Endpoint loop wall time; NOT inference health.",read(heartbeat_)/1e6);
  scalar("process_start_time_seconds","gauge","Native telemetry initialization wall time.",started_/1e6);
  gauge("initialization_state","0 loading; 1 ready; 2 stopping.",registry.initialization);
  o<<"# HELP pico:inference_progress_timestamp_seconds Last completed owner boundary; idle need not advance.\n# TYPE pico:inference_progress_timestamp_seconds gauge\n";
  const char* phases[]={"scheduler","prefill","decode"};
  for(size_t i=0;i<3;++i) o<<"pico:inference_progress_timestamp_seconds{"<<labels_<<",phase=\""<<phases[i]<<"\"} "<<read(registry.progress[i])/1e6<<'\n';
  scalar("kv_cache_usage_available","gauge","Zero: no vLLM block-pool equivalent; kv_cache_usage_perc intentionally absent.",0);
  gauge("slots_active","Rows in the current decode step; not cached sessions.",registry.slots);
  gauge("slots_capacity","Configured logical session slots.",registry.slots_capacity);
  gauge("context_capacity_tokens","Sum of configured slot context limits; not physical KV bytes.",registry.context_capacity);
  o<<"# HELP pico:session_usage_bytes Logical allocated snapshot bytes; not active KV.\n# TYPE pico:session_usage_bytes gauge\n"
   <<"# HELP pico:session_capacity_bytes Reserved snapshot capacity.\n# TYPE pico:session_capacity_bytes gauge\n";
  o<<"pico:session_usage_bytes{"<<labels_<<",tier=\"memory\"} "<<read(registry.memory_bytes)<<'\n'
   <<"pico:session_usage_bytes{"<<labels_<<",tier=\"ssd\"} "<<read(registry.ssd_bytes)<<'\n'
   <<"pico:session_capacity_bytes{"<<labels_<<",tier=\"memory\"} "<<read(registry.memory_capacity)<<'\n'
   <<"pico:session_capacity_bytes{"<<labels_<<",tier=\"ssd\"} "<<read(registry.ssd_capacity)<<'\n';
  gauge("sessions","Snapshot records across both tiers, including empty records.",registry.sessions);
  scalar("session_observed_timestamp_seconds","gauge","Last owner-published snapshot store state.",read(registry.state_observed)/1e6);
  // Logical requests exist only on rank zero. Followers must not duplicate them.
  if(rank_!=0) return o.str();
  gauge("num_requests_running","Admitted core requests, including time-sliced / awaiting DONE.",registry.running);
  gauge("num_requests_waiting","Received core submits not yet admitted; includes socket / session / slot queues.",registry.waiting);
  counter("requests_received_total","Validated socket SUBMIT frames.",registry.received);
  counter("prompt_tokens_total","Input tokens of initially admitted requests, including reused positions.",registry.prompt);
  counter("generation_tokens_total","Accepted core tokens at frontend enqueue attempt, including first and stop; no drafts.",registry.generated);
  counter("prefix_cache_queries_total","Prompt positions considered once per initial reuse plan.",registry.prompt);
  counter("prefix_cache_hits_total","Prompt positions reused by the initial plan; token-weighted, not requests.",registry.reused);
  counter("prefill_tokens_total","Logical prompt positions completed, including resume recomputation, excluding reused positions.",registry.prefilled);
  o<<"# HELP pico:request_success_total Successful terminal core requests by reason; context is separate.\n# TYPE pico:request_success_total counter\n";
  const char* reasons[]={"stop","length","context","cancelled","error","unknown"};
  for(size_t i=0;i<2;++i) o<<"pico:request_success_total{"<<labels_<<",finished_reason=\""<<reasons[i]<<"\"} "<<read(registry.outcomes[i])<<'\n';
  o<<"# HELP pico:requests_finished_total All terminal core outcomes.\n# TYPE pico:requests_finished_total counter\n";
  for(size_t i=0;i<6;++i) o<<"pico:requests_finished_total{"<<labels_<<",finished_reason=\""<<reasons[i]<<"\"} "<<read(registry.outcomes[i])<<'\n';
  auto hist=[&](const char* name,const char* help,const Histogram& h) {
    o<<"# HELP pico:"<<name<<' '<<help<<"\n# TYPE pico:"<<name<<" histogram\n";
    uint64_t count=0;
    for(size_t i=0;i<h.bins.size();++i) {
      count+=read(h.bins[i]);
      o<<"pico:"<<name<<"_bucket{"<<labels_<<",le=\"";
      if(i<bounds_us.size()) o<<bounds_us[i]/1e6; else o<<"+Inf";
      o<<"\"} "<<count<<'\n';
    }
    o<<"pico:"<<name<<"_count{"<<labels_<<"} "<<count<<'\n'
     <<"pico:"<<name<<"_sum{"<<labels_<<"} "<<read(h.sum_us)/1e6<<'\n';
  };
  hist("time_to_first_token_seconds","Socket receipt to first accepted token enqueue, including queue.",registry.ttft);
  hist("request_time_per_output_token_seconds","(DONE time - first token) / (output tokens - 1); requests with >=2 outputs.",registry.tpot);
  hist("request_queue_time_seconds","Socket receipt to first admission, including session and slot waits.",registry.queue);
  return o.str();
}
void Endpoint::run() {
  std::array<Client,4> clients;
  auto drop=[](Client& c) { if(c.fd>=0) close(c.fd); c=Client{}; };
  while(!stop_.load(std::memory_order_relaxed)) {
    set(heartbeat_, wall_us());
    std::array<pollfd,5> fds{}; fds[0]={fd_,POLLIN,0};
    for(size_t i=0;i<clients.size();++i) fds[i+1]={clients[i].fd,short(clients[i].output.empty()?POLLIN:POLLOUT),0};
    const int rc=poll(fds.data(),fds.size(),20);
    if(rc<0) { if(errno==EINTR) continue; break; }
    // Process old clients before accepting: do not mistake old poll bits for a new fd.
    for(size_t i=0;i<clients.size();++i) {
      auto& c=clients[i]; if(c.fd<0) continue;
      const auto events=fds[i+1].revents;
      if(steady_us()>c.deadline||(events&(POLLERR|POLLNVAL))) { drop(c); continue; }
      if(events&POLLIN) {
        char b[1024]; const ssize_t n=recv(c.fd,b,sizeof b,0);
        if(n==0) { drop(c); continue; }
        if(n<0) { if(errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR) drop(c); continue; }
        c.input.append(b,n);
        if(c.input.size()>4096) { drop(c); continue; }
        if(c.input.find("\r\n\r\n")!=std::string::npos) {
          const bool metrics=c.input.starts_with("GET /metrics HTTP/1.");
          const bool live=c.input.starts_with("GET /live HTTP/1.");
          std::string body=metrics?render():(live?"ok\n":"not found\n");
          if(body.size()>(256u<<10)) { drop(c); continue; }
          c.output=std::string("HTTP/1.1 ")+(metrics||live?"200 OK":"404 Not Found")+
            "\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nConnection: close\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body;
        }
      }
      if(c.fd>=0&&(events&POLLOUT)&&!c.output.empty()) {
        const ssize_t n=send(c.fd,c.output.data()+c.sent,c.output.size()-c.sent,MSG_NOSIGNAL);
        if(n>0) { c.sent+=n; if(c.sent==c.output.size()) drop(c); }
        else if(n<0&&errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR) drop(c);
      }
      if(c.fd>=0&&(events&POLLHUP)&&c.output.empty()) drop(c);
    }
    if(fds[0].revents&POLLIN) {
      // Bound accepts per poll iteration even under an adversarial connection flood.
      for(size_t j=0;j<4;++j) {
        const int fd=accept4(fd_,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);
        if(fd<0) break;
        auto it=std::find_if(clients.begin(),clients.end(),[](auto& c){return c.fd<0;});
        if(it==clients.end()) close(fd);
        else { it->fd=fd; it->deadline=steady_us()+1000000; }
      }
    }
  }
  for(auto& c:clients) drop(c);
}
} // namespace pico::metrics
