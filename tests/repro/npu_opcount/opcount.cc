// Re-derive OP_COUNT at the SHIPPED 8x8 array, against a prediction made first.
//
// WHY THIS EXISTS. Our twenty OP_COUNT measurements were taken at NUM_ROWS =
// NUM_COLS = 4. GRX930 then widened the array to 8x8 (06d82fd) and confirmed
// 8x8 as the shipped default (9f3c2b3), so we invalidated our own table -- see
// cuda_mapping.md 7.34. This re-derives it and, more usefully, tests an
// explanation of it.
//
// THE PREDICTION, WRITTEN BEFORE THE RUN. At 4x4 the measured relation was
//
//     OP_COUNT = 10 * M * ceil(N/NUM_COLS) * ceil(K/NUM_ROWS) * NUM_ROWS * NUM_COLS
//
// and the 10 was unexplained -- a constant that fitted twenty points. GRX930's
// new timing note gives the S_RUN pass length as NUM_ROWS + NUM_COLS + 2,
// which is exactly 10 at 4x4. If that is what our constant was, then at 8x8 it
// must be 18, and the whole relation is parameterized rather than fitted.
//
// Three hypotheses, well separated at 8x8 (MAX_M=8, so ceil(M/8)==1 for every
// legal M -- which means their published form drops M entirely):
//
//   H1  ours, parameterized: (R+C+2) * M * ceil(N/C) * ceil(K/R) * R * C
//   H2  the constant is literally 10:  10 * M * ceil(N/C) * ceil(K/R) * R * C
//   H3  theirs, published:  ceil(M/R) * ceil(N'/C) * ceil(K/R) * R * C
//
// H1 and H2 differ by 1.8x on every shape; H3 differs from both by the whole
// factor of M. One run separates all three.
//
// CALIBRATE FIRST. `--calibrate` runs ONE shape three times on one instance
// with no reset between. If OP_COUNT climbs, it is cumulative and every number
// below is the sum of the run so far rather than the shape's cost. This check
// is here because the same class of error -- a counter that does not restart
// where the harness assumed it did -- cost this project a roadmap phase.
#include "Vc930_npu_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdlib>

namespace {
constexpr uint32_t kDdrBytes = 65536;

struct Rtl {
  Vc930_npu_top top;
  std::vector<uint8_t> ddr = std::vector<uint8_t>(kDdrBytes, 0);
  uint64_t cycles = 0;
  bool r_active=false; uint32_t r_addr=0; int r_beats=0, r_size=8;
  bool w_active=false; uint32_t w_addr=0; bool b_pending=false;

  void tick(){ top.i_clk=0; top.eval(); service(); top.i_clk=1; top.eval(); ++cycles; }
  void service(){
    top.m_axi_arready = !r_active;
    if (top.m_axi_arvalid && !r_active){ r_active=true; r_addr=top.m_axi_araddr;
      r_beats=top.m_axi_arlen+1; r_size=1<<top.m_axi_arsize; }
    if (r_active){
      uint64_t d=0;
      for(int i=0;i<8;++i){ uint32_t a=r_addr+i; if(a<kDdrBytes) d|=(uint64_t)ddr[a]<<(8*i); }
      top.m_axi_rdata=d; top.m_axi_rresp=0; top.m_axi_rvalid=1; top.m_axi_rlast=(r_beats==1);
      if(top.m_axi_rready){ r_addr+=r_size; if(--r_beats==0) r_active=false; }
    } else { top.m_axi_rvalid=0; top.m_axi_rlast=0; }
    top.m_axi_awready = !w_active;
    if (top.m_axi_awvalid && !w_active){ w_active=true; w_addr=top.m_axi_awaddr; }
    top.m_axi_wready = w_active;
    if (w_active && top.m_axi_wvalid){
      for(int i=0;i<8;++i){ if(!((top.m_axi_wstrb>>i)&1)) continue;
        uint32_t a=w_addr+i; if(a<kDdrBytes) ddr[a]=(uint8_t)(top.m_axi_wdata>>(8*i)); }
      w_addr+=8; if(top.m_axi_wlast){ w_active=false; b_pending=true; }
    }
    top.m_axi_bvalid=b_pending; top.m_axi_bresp=0;
    if(b_pending && top.m_axi_bready) b_pending=false;
  }
  void reset(){
    top.i_rst_n=0;
    top.s_axi_awvalid=top.s_axi_wvalid=top.s_axi_bready=0;
    top.s_axi_arvalid=top.s_axi_rready=0;
    top.m_axi_arready=top.m_axi_rvalid=top.m_axi_rlast=0;
    top.m_axi_awready=top.m_axi_wready=top.m_axi_bvalid=0;
    for(int i=0;i<16;++i) tick();
    top.i_rst_n=1;
    for(int i=0;i<4;++i) tick();
  }
  void csr_write(uint32_t off,uint32_t v){
    top.s_axi_awaddr=off; top.s_axi_awvalid=1; top.s_axi_wdata=v;
    top.s_axi_wstrb=0xF; top.s_axi_wvalid=1; top.s_axi_bready=1;
    bool aw=false,w=false,b=false;
    for(int i=0;i<64 && !(aw&&w&&b);++i){ tick();
      if(top.s_axi_awready){ top.s_axi_awvalid=0; aw=true; }
      if(top.s_axi_wready){ top.s_axi_wvalid=0; w=true; }
      if(top.s_axi_bvalid) b=true; }
    top.s_axi_awvalid=top.s_axi_wvalid=0; tick(); top.s_axi_bready=0;
  }
  uint32_t csr_read(uint32_t off){
    top.s_axi_araddr=off; top.s_axi_arvalid=1; top.s_axi_rready=1;
    uint32_t got=0;
    for(int i=0;i<64;++i){ tick();
      if(top.s_axi_arready) top.s_axi_arvalid=0;
      if(top.s_axi_rvalid){ got=top.s_axi_rdata; break; } }
    top.s_axi_arvalid=0; tick(); top.s_axi_rready=0; return got;
  }
  void program(int m,int n,int k,uint32_t A,uint32_t B,uint32_t C){
    csr_write(0x08,m); csr_write(0x0C,n); csr_write(0x10,k);
    csr_write(0x14,A); csr_write(0x18,B); csr_write(0x1C,C); csr_write(0x20,0);
  }
};

int32_t ref_gemm(const int8_t*A,const int8_t*B,int n,int k,int i,int j){
  int32_t s=0; for(int p=0;p<k;++p) s+=(int32_t)A[i*k+p]*(int32_t)B[p*n+j]; return s;
}
}  // namespace

namespace {
constexpr int R = NUM_ROWS_P, C = NUM_COLS_P;   // -D from the build
inline int ceil_div(int a,int b){ return (a+b-1)/b; }

// The three predictions. Kept as functions so the table below prints all of
// them and the reader picks the winner, rather than the harness declaring one.
uint64_t h1(int m,int n,int k){ return (uint64_t)(R+C+2)*m*ceil_div(n,C)*ceil_div(k,R)*R*C; }
uint64_t h2(int m,int n,int k){ return (uint64_t)10        *m*ceil_div(n,C)*ceil_div(k,R)*R*C; }
uint64_t h3(int m,int n,int k){ return (uint64_t)ceil_div(m,R)*ceil_div(n,C)*ceil_div(k,R)*R*C; }

// GRX930's CYCLE model, from the same doc that gave us R+C+2 (9f3c2b3):
//   core = M x [ (K+1)*N + ceil(K/R)*(R+C+2)*(number of N-tiles) ]
// Tested here because a team whose OP_COUNT formula misses every shape may
// still have the cycle model right, and lumping the two together would be
// unfair to them and useless to us.
uint64_t hcyc(int m,int n,int k){
  return (uint64_t)m * ((uint64_t)(k+1)*n + (uint64_t)ceil_div(k,R)*(R+C+2)*ceil_div(n,C));
}

struct Res { uint32_t cyc, ops, stall, dma; int bad; bool no_done; };

// One GEMM on a freshly reset instance. Sequential path only: write CSRs,
// START, poll !BUSY, require DONE -- the contract's documented-safe single
// command case, so nothing here depends on the queue.
Res run_one(Rtl& r, int m,int n,int k, bool do_reset) {
  if (do_reset) r.reset();
  const uint32_t A=0x0100, B=0x0800, Cb=0x1000;
  std::vector<int8_t> As((size_t)m*k), Bs((size_t)k*n);
  for(size_t i=0;i<As.size();++i) As[i]=(int8_t)((i*7+1)%13-6);
  for(size_t i=0;i<Bs.size();++i) Bs[i]=(int8_t)((i*5+3)%11-5);
  for(size_t i=0;i<As.size();++i) r.ddr[A+i]=(uint8_t)As[i];
  for(size_t i=0;i<Bs.size();++i) r.ddr[B+i]=(uint8_t)Bs[i];
  std::memset(&r.ddr[Cb],0,(size_t)m*n*4);

  r.program(m,n,k,A,B,Cb);
  r.csr_write(0x00,1);                       // CTRL.START (also clears DONE)

  // WAIT FOR BUSY TO RISE BEFORE WAITING FOR IT TO FALL.
  //
  // The first version polled only for !BUSY. Right after a START the engine
  // has not asserted BUSY yet, so that loop exited on its first read and
  // sampled the counters before anything had happened: the calibration run
  // reported OP_COUNT=0, CYCLE_LO=0 and wrong answers for the FIRST command
  // after reset, and correct numbers for every command after a completed one
  // (where BUSY was still high on entry). One shape in fifteen would have been
  // silently zero.
  //
  // This is the same defect, in the same shape, as the rtlsim run() bug this
  // project reported to grxgpu -- sample a level signal before its transient
  // and you measure the wrong side of it. Bounded, because a legal command
  // that completes very fast might never be observed busy, and that must not
  // hang.
  bool seen_busy=false;
  for(int p=0;p<2000 && !seen_busy;++p) seen_busy = (r.csr_read(0x04)&0x1)!=0;
  for(int p=0;p<400000;++p){ const uint32_t st=r.csr_read(0x04);
    if(!(st&0x1)) break; }                   // !BUSY
  const bool done = (r.csr_read(0x04)&0x2)!=0;
  Res out{};
  out.no_done = !done;
  out.cyc  = r.csr_read(0x24);
  out.ops  = r.csr_read(0x2c);
  out.stall= r.csr_read(0x30);
  out.dma  = r.csr_read(0x34);
  for(int i=0;i<m;++i) for(int j=0;j<n;++j){
    const uint32_t a=Cb+((size_t)i*n+j)*4;
    int32_t got=(int32_t)(r.ddr[a]|(r.ddr[a+1]<<8)|(r.ddr[a+2]<<16)|((uint32_t)r.ddr[a+3]<<24));
    if(got!=ref_gemm(As.data(),Bs.data(),n,k,i,j)) ++out.bad;
  }
  return out;
}
}  // namespace

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  const bool calibrate = (argc>1 && std::strcmp(argv[1],"--calibrate")==0);
  std::printf("c930 NPU OP_COUNT, array %dx%d\n\n", R, C);

  if (calibrate) {
    // Same shape, three times, ONE reset. A per-command counter reports the
    // same number three times; a cumulative one climbs.
    Rtl r; r.reset();
    std::printf("%-6s %10s %10s   same shape, no reset between\n","run","OP_COUNT","CYCLE_LO");
    for(int i=0;i<3;++i){ Res x=run_one(r,4,4,4,false);
      std::printf("%-6d %10u %10u%s\n", i, x.ops, x.cyc, (x.bad||x.no_done)?"  WRONG/NO-DONE":""); }
    std::printf("\nIf OP_COUNT climbs, it is cumulative and the sweep below is\n"
                "the sum of the run rather than the shape. Reported either way.\n");
    return 0;
  }

  struct Sh { int m,n,k; };
  const Sh shapes[] = {
    {1,1,1},{1,4,4},{2,4,4},{4,4,4},{8,4,4},
    {3,4,4},{5,4,4},{6,4,4},{7,8,4},{8,12,5},
    {4,8,4},{4,12,4},{4,4,8},{4,4,16},{8,12,16},
  };
  const int NS = (int)(sizeof(shapes)/sizeof(shapes[0]));
  int hit1=0,hit2=0,hit3=0,hitc=0,wrong=0;

  std::printf("%-12s %10s %10s %10s %10s %8s %8s\n",
              "shape","OP_COUNT","H1 (R+C+2)","H2 (10)","H3 theirs","CYCLE","cyc model");
  Rtl r;
  for(int s=0;s<NS;++s){
    const Sh& q=shapes[s];
    Res x = run_one(r,q.m,q.n,q.k,true);     // fresh reset per shape
    const uint64_t p1=h1(q.m,q.n,q.k), p2=h2(q.m,q.n,q.k), p3=h3(q.m,q.n,q.k);
    if(x.ops==p1) ++hit1; if(x.ops==p2) ++hit2; if(x.ops==p3) ++hit3;
    const uint64_t pc=hcyc(q.m,q.n,q.k); if(x.cyc==pc) ++hitc;
    if(x.bad) ++wrong;
    char sh[16]; std::snprintf(sh,sizeof sh,"%dx%dx%d",q.m,q.n,q.k);
    std::printf("%-12s %10u %10llu%s %9llu%s %9llu%s %8u %8u%s%s\n", sh, x.ops,
                (unsigned long long)p1, x.ops==p1?"*":" ",
                (unsigned long long)p2, x.ops==p2?"*":" ",
                (unsigned long long)p3, x.ops==p3?"*":" ",
                x.cyc, (unsigned)pc, x.cyc==pc?"*":" ",
                (x.bad||x.no_done)?(x.no_done?"  NO DONE":"  WRONG"):"");
  }
  std::printf("\nOP_COUNT:  H1 (R+C+2 = %d) %d/%d   H2 (10) %d/%d   H3 (theirs) %d/%d\n",
              R+C+2, hit1,NS, hit2,NS, hit3,NS);
  std::printf("CYCLE:     their published timing model %d/%d\n", hitc,NS);
  std::printf("%d of %d shapes computed a wrong answer.\n", wrong, NS);
  return wrong ? 1 : 0;
}
