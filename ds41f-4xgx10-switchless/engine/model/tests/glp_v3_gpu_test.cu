// Synthetic activation conformance only. Run inside an authorized stopped-ring window.
// Compares the actual CUDA kernels to an independent host transcription of
// the retained TileLang source, with the real GLP-27 directions (no model prompts).
#include "model/glp.h"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
static uint32_t rng=27112026;
static uint32_t next(){return rng=rng*1664525u+1013904223u;}
static float widen(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;memcpy(&f,&u,4);return f;}
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static uint16_t value(){return uint16_t(((next()>>31)<<15)|(uint32_t(123+next()%9)<<7)|(next()%128));}
static float add(float a,float b){volatile float v=a+b;return v;}
static float mul(float a,float b){volatile float v=a*b;return v;}
static float dot(const float*h,const float*d){
 std::array<float,320>a{};
 for(int t=0;t<320;++t)for(int j=0;j<16;++j){int c=(j/8*320+t)*8+j%8;a[t]=std::fma(h[c],d[c],a[t]);}
 for(int off=16;off;off/=2){auto old=a;for(int t=0;t<320;++t)a[t]=add(old[t],old[t^off]);}
 float total=0;for(int w=0;w<10;++w)total=add(total,a[32*w]);return total;
}
template<class T> struct Buf{T*p;Buf(size_t n){CK(cudaMalloc(&p,n*sizeof(T)));}~Buf(){cudaFree(p);}};
template<class T>static void put(T*p,const std::vector<T>&h){CK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
int main(int argc,char**argv){
 if(argc!=2)return 2;auto file=model::detail::glp_read(argv[1],40,5120);
 if(file.hook!="hc_fold_pre_engram")return 2;
 size_t bad=0,checked=0;
 for(int R:{1,4,16,65})for(int L:{11,23,37}){
 std::vector<uint16_t> wr(R*5120),res(R*4*5120),out(res.size()),want(res.size()),gotwrite(wr.size());
 std::vector<float> post(R*4),comb(R*16),dir(file.dir.begin()+L*5120,file.dir.begin()+(L+1)*5120),alpha{3.5f};
 for(auto&v:wr)v=value();for(auto&v:res)v=value();for(auto&v:post)v=float(next()%100001)/100000.f;for(auto&v:comb)v=float(next()%100001)/100000.f;
 Buf<uint16_t>w(wr.size()),r(res.size()),o(res.size());Buf<float>d(5120),a(1),p(post.size()),c(comb.size());
 put(w.p,wr);put(r.p,res);put(d.p,dir);put(a.p,alpha);put(p.p,post);put(c.p,comb);
 model::ops::glp_fold_steer(o.p,R,d.p,a.p,w.p,r.p,p.p,c.p,nullptr,nullptr,0,true);
 CK(cudaDeviceSynchronize());CK(cudaMemcpy(out.data(),o.p,out.size()*2,cudaMemcpyDeviceToHost));
 for(int t=0;t<R;++t){
 std::vector<float> h(4*5120),mean(5120);
 for(int col=0;col<5120;++col){
  for(int n=0;n<4;++n){
   float v=mul(comb[t*16+n],widen(res[t*4*5120+col]));v=std::fma(post[t*4+n],widen(wr[t*5120+col]),v);
   for(int m=1;m<4;++m)v=std::fma(comb[t*16+m*4+n],widen(res[(t*4+m)*5120+col]),v);
   h[n*5120+col]=widen(bf(v));
  }
  mean[col]=mul(add(add(add(h[col],h[5120+col]),h[10240+col]),h[15360+col]),.25f);
 }
 float scale=mul(3.5f,dot(mean.data(),dir.data()));
 for(int n=0;n<4;++n)for(int col=0;col<5120;++col)want[(t*4+n)*5120+col]=bf(add(h[n*5120+col],-mul(scale,dir[col])));
 }
 for(size_t i=0;i<out.size();++i){bad+=out[i]!=want[i];++checked;}
 model::ops::glp_steer_write(w.p,R,d.p,a.p,0);CK(cudaDeviceSynchronize());CK(cudaMemcpy(gotwrite.data(),w.p,wr.size()*2,cudaMemcpyDeviceToHost));
 for(int t=0;t<R;++t){std::vector<float>h(5120);for(int col=0;col<5120;++col)h[col]=widen(wr[t*5120+col]);float scale=mul(3.5f,dot(h.data(),dir.data()));for(int col=0;col<5120;++col){bad+=gotwrite[t*5120+col]!=bf(add(h[col],-mul(scale,dir[col])));++checked;}}
 }
 printf("{\"gpu_execution\":true,\"rows\":[1,4,16,65],\"layers\":[11,23,37],\"bf16_values_checked\":%zu,\"differences\":%zu}\n",checked,bad);
 return bad?1:0;
}
