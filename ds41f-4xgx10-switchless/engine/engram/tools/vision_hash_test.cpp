// Host contract: negative image IDs must exactly match the reference's live=0 mask,
// including every crossing n-gram and chunk start. No CUDA execution claimed.
#include "engram/engram.h"
#include <cassert>
#include <cstdio>
#include <vector>
int main() {
 using namespace engram;
 HashMap m; m.pad=0;m.n_tokens=m.vocab=32;
 for(int i=0;i<32;++i)m.token_map.push_back(i);
 for(int l=0;l<kLayers;++l){
  for(int s=0;s<kMaxNgram;++s)m.mult[l][s]=3+2*(s+l);
  for(int o=0;o<kOrders;++o)for(int h=0;h<kHeads;++h){m.prime[l][o][h]=101+2*h+o*20;m.offset[l][o*kHeads+h]=o*1000+h*200;}
 }
 size_t checks=0;
 for(int len=1;len<64;++len)for(int lo=0;lo<len;++lo) {
  std::vector<int32_t> ids(len),neg(len);std::vector<uint8_t> live(len,1);
  for(int i=0;i<len;++i)ids[i]=neg[i]=i%31+1;
  for(int i=lo;i<std::min(len,lo+5);++i){live[i]=0;neg[i]=-1;}
  for(int start=0;start<len;++start){
   std::vector<uint32_t>a((len-start)*kLayers*kCols),b(a.size());
   hash_positions(m,ids.data(),live.data(),start,len-start,a.data());
   hash_positions(m,neg.data(),nullptr,start,len-start,b.data());assert(a==b);++checks;
  }
 }
 printf("PASS: %zu image-mask/crossing-ngram/chunk-start comparisons; host only\n",checks);
}
