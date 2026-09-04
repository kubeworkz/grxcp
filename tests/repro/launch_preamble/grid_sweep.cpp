#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>
#include "grx/grx_runtime.h"
struct pargs { uint32_t abi; uint32_t slots; uint64_t out; };
int main(int argc, char** argv) {
  const bool desc = (argc > 1 && argv[1][0]=='d');
  grxDeviceProp_t p{}; grxGetDeviceProperties(&p, 0);
  std::printf("%s: %d SMs\n\n", p.name, p.multiProcessorCount);
  grxModule_t m=nullptr; if (grxModuleLoad(&m,"build-real/preamble.vxbin")!=grxSuccess) return 1;
  grxFunction_t f=nullptr; if (grxModuleGetFunction(&f,m,"preamble_probe")!=grxSuccess) return 1;
  const int MAXB=64; const size_t bytes=(size_t)MAXB*4*8;
  void* d=nullptr; if (grxMalloc(&d,bytes)!=grxSuccess) return 1;
  std::printf("%-8s %12s %12s %12s %10s\n","blocks","first entry","last entry","spread","cores");
  int order_a[7]={1,2,4,8,16,32,64}, order_d[7]={64,32,16,8,4,2,1};
  for (int oi=0; oi<7; ++oi) { int nb = desc ? order_d[oi] : order_a[oi];
    grxMemset(d,0,bytes);
    pargs a{}; a.abi=3; a.slots=MAXB; a.out=(uint64_t)(uintptr_t)d;
    grxError_t e=grxLaunchFunction(f,dim3_t{(unsigned)nb,1,1},dim3_t{16,1,1},&a,sizeof(a),0,nullptr);
    if (e==grxSuccess) e=grxDeviceSynchronize();
    std::vector<uint64_t> o((size_t)MAXB*4,0);
    if (e==grxSuccess) grxMemcpy(o.data(),d,bytes,grxMemcpyDefault);
    uint64_t lo=~0ull, hi=0; int seen=0; std::vector<int> cores;
    for (int b=0;b<nb;++b){ uint64_t t=o[(size_t)b*4]; uint64_t c=o[(size_t)b*4+3];
      if (c==0) continue; ++seen; lo=std::min(lo,t); hi=std::max(hi,t);
      if (std::find(cores.begin(),cores.end(),(int)c-1)==cores.end()) cores.push_back((int)c-1); }
    if (!seen) { std::printf("%-8d %12s\n", nb, "(none)"); continue; }
    std::printf("%-8d %12llu %12llu %12llu %10zu\n", nb,
                (unsigned long long)lo,(unsigned long long)hi,
                (unsigned long long)(hi-lo), cores.size());
  }
  std::printf("\nfirst entry = earliest block reaching its first instruction\n");
  std::printf("If FIRST stays flat while LAST grows, dispatch overlaps work.\n");
  std::printf("If FIRST grows with the grid, nothing runs until dispatch is done.\n");
  return 0;
}
