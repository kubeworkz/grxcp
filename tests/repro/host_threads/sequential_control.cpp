// CONTROL: does any of this fail with ONE thread? If loading the same module
// twice fails sequentially, stage 2 is not a threading finding at all.
#include <cstdio>
#include "grx/grx_runtime.h"
int main(){
  grxDeviceProp_t p{}; grxGetDeviceProperties(&p,0);
  std::printf("%s\n\n", p.name);
  std::printf("-- load the same module twice, sequentially, both still open --\n");
  grxModule_t a=nullptr,b=nullptr;
  grxError_t ra=grxModuleLoad(&a,"build-real/preamble.vxbin");
  grxError_t rb=grxModuleLoad(&b,"build-real/preamble.vxbin");
  std::printf("   first=%d second=%d\n",(int)ra,(int)rb);
  if(a)grxModuleUnload(a); if(b)grxModuleUnload(b);
  std::printf("-- load / unload / load, sequentially --\n");
  grxModule_t c=nullptr; grxError_t rc=grxModuleLoad(&c,"build-real/preamble.vxbin");
  if(c)grxModuleUnload(c);
  grxModule_t d=nullptr; grxError_t rd=grxModuleLoad(&d,"build-real/preamble.vxbin");
  std::printf("   first=%d after-unload=%d\n",(int)rc,(int)rd);
  if(d)grxModuleUnload(d);
  std::printf("-- events, sequentially --\n");
  int bad=0;
  for(int i=0;i<40;++i){ grxEvent_t e=nullptr;
    if(grxEventCreate(&e)!=grxSuccess){++bad;continue;}
    grxEventRecord(e,nullptr); grxEventQuery(e);
    if(grxEventDestroy(e)!=grxSuccess)++bad; }
  std::printf("   %d failures in 40\n",bad);
  return 0;
}
