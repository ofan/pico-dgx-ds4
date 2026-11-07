// Host-only spec-v3 arithmetic conformance, no model/evaluation prompts.
// Oracle transcribes the retained reference tilelang/glp.py; candidate uses
// the same fold/mean/layout helpers as the device kernels. Real GLP-27 required.
#include "model/glp.h"
#include "model/glp_math.h"
#include <array>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>
using namespace model::detail;
static uint32_t rng=27112026;
static uint32_t next(){return rng=rng*1664525u+1013904223u;}
static float widen(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
static uint16_t bf(float f){uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static float value(){return widen(uint16_t(((next()>>31)<<15)|(uint32_t(123+next()%9)<<7)|(next()%128)));}
static float add(float a,float b){volatile float v=a+b;return v;}
static float mul(float a,float b){volatile float v=a*b;return v;}
static float oracle_fold(float x,const float* r,float p,const float* c,int n){
 float v=mul(c[n],r[0]);v=std::fma(p,x,v);
 for(int m=1;m<4;++m)v=std::fma(c[4*m+n],r[m],v);return widen(bf(v));
}
static float dot(const float* h,const float* d,bool candidate){
 std::array<float,320>a{};
 for(int t=0;t<320;++t)for(int j=0;j<16;++j){
  int c=candidate?glp_column(t,j):(j/8*320+t)*8+j%8;
  a[t]=candidate?glp_fma(h[c],d[c],a[t]):std::fma(h[c],d[c],a[t]);
 }
 for(int off=16;off;off/=2){auto old=a;for(int t=0;t<320;++t)a[t]=candidate?glp_add(old[t],old[t^off]):add(old[t],old[t^off]);}
 float total=0;for(int w=0;w<10;++w)total=candidate?glp_add(total,a[32*w]):add(total,a[32*w]);return total;
}
int main(int argc,char**argv){
 if(argc<2)return 2;
 auto file=glp_read(argv[1],40,5120);
 assert(file.hook=="hc_fold_pre_engram" && file.alpha==3.5f && file.layers.size()==27);
 assert(file.layers.front()==11 && file.layers.back()==37);
 if(argc==3){FILE*f=fopen(argv[2],"wb");assert(f);assert(fwrite(file.dir.data(),sizeof(float),file.dir.size(),f)==file.dir.size());fclose(f);}
 size_t folds=0,means=0,dots=0,outputs=0,bad=0,non_equivalent=0;
 std::vector<float> mean(5120),want_mean(5120),write(5120),h(4*5120);
 for(int layer:file.layers)for(int row=0;row<4;++row){
  const float*d=file.dir.data()+layer*5120;
  for(int c=0;c<5120;++c){
   float part[4],r[4],comb[16],post[4];
   for(auto&v:part)v=value();for(auto&v:r)v=value();
   for(auto&v:comb)v=float(next()%100001)/100000.f;for(auto&v:post)v=float(next()%100001)/100000.f;
   // Rank-order FP32 sum, one BF16 cast (collectives already provide this).
   float x=part[0];for(int k=1;k<4;++k)x=add(x,part[k]);write[c]=widen(bf(x));
   float streams[4];
   for(int n=0;n<4;++n){
    float want=oracle_fold(write[c],r,post[n],comb,n);
    streams[n]=widen(bf(glp_fold(write[c],r,post[n],comb,n)));
    bad+=bf(want)!=bf(streams[n]);++folds;h[n*5120+c]=streams[n];
   }
   want_mean[c]=mul(add(add(add(streams[0],streams[1]),streams[2]),streams[3]),.25f);
   mean[c]=glp_mean4(streams[0],streams[1],streams[2],streams[3]);
   bad+=memcmp(&want_mean[c],&mean[c],4)!=0;++means;
  }
  float want=dot(want_mean.data(),d,false),got=dot(mean.data(),d,true);
  bad+=memcmp(&want,&got,4)!=0;++dots;
  const float scale=glp_mul(file.alpha,got),want_scale=mul(file.alpha,want);
  for(int n=0;n<4;++n){
   float v1scale=mul(file.alpha,dot(h.data()+n*5120,d,false));
   for(int c=0;c<5120;++c){
    auto out=bf(glp_add(h[n*5120+c],-glp_mul(scale,d[c])));
    auto expected=bf(add(h[n*5120+c],-mul(want_scale,d[c])));
    bad+=out!=expected;++outputs;
    non_equivalent+=out!=bf(add(h[n*5120+c],-mul(v1scale,d[c])));
   }
  }
  // Companion write: same 320x16 projection, independent of streams.
  want=dot(write.data(),d,false);got=dot(write.data(),d,true);
  bad+=memcmp(&want,&got,4)!=0;++dots;
  for(int c=0;c<5120;++c){
   bad+=bf(glp_add(write[c],-glp_mul(glp_mul(file.alpha,got),d[c])))!=bf(add(write[c],-mul(mul(file.alpha,want),d[c])));++outputs;
  }
 }
 printf("{\"layers\":27,\"rows_per_layer\":4,\"fold_values\":%zu,\"mean_values\":%zu,\"dot_rows\":%zu,\"projected_values\":%zu,\"host_differences\":%zu,\"v1_vs_v3_differences\":%zu,\"gpu_execution\":false}\n",folds,means,dots,outputs,bad,non_equivalent);
 return bad||!non_equivalent;
}
