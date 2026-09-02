// Is dev_exp_nonpos bit-identical to dev_exp for every x <= 0?
//
// The claim in dnn_device.h is that this is an exact specialisation, not an
// approximation. That is a claim about float arithmetic, so it is checked
// against float arithmetic rather than argued from the algebra.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
static float fmin_(float a,float b){return __builtin_fminf(a,b);}
static float fmax_(float a,float b){return __builtin_fmaxf(a,b);}
static float csgn_(float m,float s){return __builtin_copysignf(m,s);}

static float dev_exp(float x){
  x = fmin_(fmax_(x,-88.0f),88.0f);
  const float A=1.44269504088896340736f,H=0.693359375f,L=-2.12194440e-4f;
  const int k=(int)(x*A+csgn_(0.5f,x));
  const float r=(x-(float)k*H)-(float)k*L;
  const float r2=r*r;
  const float p=1.0f+r+r2*(0.5f+r*(0.16666666666f+r*(0.04166666666f+r*0.00833333333f)));
  union{float f;uint32_t u;}s; s.u=(uint32_t)((k+127)&0xFF)<<23;
  return p*s.f;
}
static float dev_exp_nonpos(float x){
  x = fmax_(x,-88.0f);
  const float A=1.44269504088896340736f,H=0.693359375f,L=-2.12194440e-4f;
  const int k=(int)(x*A-0.5f);
  const float r=(x-(float)k*H)-(float)k*L;
  const float r2=r*r;
  const float p=1.0f+r+r2*(0.5f+r*(0.16666666666f+r*(0.04166666666f+r*0.00833333333f)));
  union{float f;uint32_t u;}s; s.u=(uint32_t)(k+127)<<23;
  return p*s.f;
}
static bool same(float a,float b){uint32_t x,y;std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);return x==y;}

int main(){
  long long n=0, bad=0; float worst=0;
  // Every representable negative float down to -200, by bit pattern, plus the
  // edges: -0.0, the clamp, and past it.
  for (uint32_t u=0x80000000u; u<0xFF000000u; u+=1u<<7) {
    float x; std::memcpy(&x,&u,4);
    if (!(x<=0.0f)) continue;
    if (x < -200.0f) continue;
    ++n;
    const float a=dev_exp(x), b=dev_exp_nonpos(x);
    if(!same(a,b)){ if(bad<5) std::printf("  MISMATCH x=%.9g  dev_exp=%.9g  nonpos=%.9g\n",x,a,b); ++bad; }
  }
  const float edges[]={0.0f,-0.0f,-1e-45f,-88.0f,-87.9999f,-88.0001f,-1e30f,-3.402823466e+38f,
                       -0.693147f,-0.3465f,-0.3466f,-1.0f,-127.0f,-200.0f};
  for(float x:edges){++n; const float a=dev_exp(x),b=dev_exp_nonpos(x);
    if(!same(a,b)){std::printf("  EDGE MISMATCH x=%.9g  %.9g vs %.9g\n",x,a,b);++bad;}}
  const float nan=std::nanf("");
  const float a=dev_exp(nan), b=dev_exp_nonpos(nan);
  std::printf("  NaN: dev_exp=%.9g  nonpos=%.9g  %s\n", a, b, same(a,b)?"identical":"*** DIFFER ***");
  if(!same(a,b)) ++bad;
  (void)worst;
  std::printf("%lld values compared, %lld mismatches\n", n, bad);
  std::printf("%s\n", bad ? "NOT an exact specialisation" : "EXACT: bit-identical for every x <= 0 and for NaN");
  return bad?1:0;
}
