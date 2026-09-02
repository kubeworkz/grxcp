// Back-to-back GEMMs on ONE c930_npu_top instance, no reset between them.
//
// GRX930 added a 4-entry command queue (243cbc9) and then fixed a race in it
// (3df215b), reporting that "the grxcp team can now launch back-to-back CTAs
// without deadlock". This is that claim, run rather than accepted.
//
// The pattern is the one the fix is about: write the CSRs, pulse START, and
// pulse START again for the next GEMM WITHOUT waiting for DONE -- so the
// second START can land while the dispatcher is still in D_WAIT, which is the
// window the race lived in. A run that never sees DONE for both is the
// deadlock.
#include "Vc930_npu_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
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

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  const int NOPS = (argc>1)?atoi(argv[1]):4;
  const int GAP  = (argc>2)?atoi(argv[2]):0;   // idle cycles between submits
  const int SEQ  = (argc>3)?atoi(argv[3]):0;   // 1 = wait for DONE between submits (control)
  struct Sh { int m,n,k; uint32_t A,B,C; };
  Sh ops[4] = {
    {4,4,4, 0x0100,0x0400,0x0800},
    {4,4,4, 0x0140,0x0440,0x0900},
    {2,4,8, 0x0180,0x0480,0x0A00},
    {8,4,4, 0x01C0,0x04C0,0x0B00},
  };
  Rtl r; r.reset();
  std::printf("=== back-to-back: %d GEMMs, one instance, ONE reset ===\n", NOPS);

  std::vector<std::vector<int8_t>> As(NOPS), Bs(NOPS);
  for(int t=0;t<NOPS;++t){
    As[t].resize(ops[t].m*ops[t].k); Bs[t].resize(ops[t].k*ops[t].n);
    for(size_t i=0;i<As[t].size();++i) As[t][i]=(int8_t)(((i*7+t)%11)-5);
    for(size_t i=0;i<Bs[t].size();++i) Bs[t][i]=(int8_t)(((i*5+t)%9)-4);
    for(size_t i=0;i<As[t].size();++i) r.ddr[ops[t].A+i]=(uint8_t)As[t][i];
    for(size_t i=0;i<Bs[t].size();++i) r.ddr[ops[t].B+i]=(uint8_t)Bs[t][i];
  }

  // Fire every START without waiting for DONE -- the queue's whole purpose,
  // and the window the race lived in.
  for(int t=0;t<NOPS;++t){
    r.program(ops[t].m,ops[t].n,ops[t].k,ops[t].A,ops[t].B,ops[t].C);
    r.csr_write(0x00,1);
    for(int g=0;g<GAP;++g) r.tick();
    if(SEQ){
      // CONTROL: the one-at-a-time protocol that worked before the queue
      // existed. If this passes and the pipelined one does not, the harness
      // drives the engine correctly and only the queued path is at fault.
      bool got=false;
      for(int p=0;p<200000 && !got;++p){ const uint32_t st=r.csr_read(0x04);
        if(st&0x2) got=true; if(st&0x4){ std::printf("  op %d ERROR\n",t); break; } }
      std::printf("    (sequential) op %d %s at cycle %llu\n", t,
                  got?"completed":"*** NEVER COMPLETED ***",(unsigned long long)r.cycles);
    }
    std::printf("  submitted op %d (%dx%dx%d)  QUEUE_STATUS=0x%x  STATUS=0x%x\n",
                t, ops[t].m,ops[t].n,ops[t].k, r.csr_read(0x38), r.csr_read(0x04));
  }

  // DONE is a LATCH cleared only by writing CTRL with bit0 set, so a latched
  // bit cannot be counted four times -- the first version of this harness did
  // exactly that and reported four completions 4 cycles apart, which is how
  // the mistake surfaced. The queue draining is the signal that does not
  // depend on DONE semantics: occupancy back to zero with BUSY clear.
  bool drained=false;
  uint64_t deadline=r.cycles;
  for(int p=0;p<400000 && !drained;++p){
    const uint32_t q=r.csr_read(0x38), st=r.csr_read(0x04);
    if(st&0x4){ std::printf("  *** ERROR bit set (STATUS=0x%x) ***\n",st); break; }
    if(((q&0xF)==0) && !(st&0x1)) drained=true;
    deadline=r.cycles;
  }
  std::printf("  final QUEUE_STATUS=0x%x  STATUS=0x%x  at cycle %llu\n",
              r.csr_read(0x38), r.csr_read(0x04), (unsigned long long)deadline);
  if(!drained){
    std::printf("\n*** DEADLOCK: the queue never drained ***\n");
    return 1;
  }
  std::printf("\nqueue drained after %llu cycles\n",(unsigned long long)r.cycles);

  int bad=0;
  for(int t=0;t<NOPS;++t)
    for(int i=0;i<ops[t].m;++i)
      for(int j=0;j<ops[t].n;++j){
        const uint32_t a=ops[t].C+(i*ops[t].n+j)*4;
        int32_t got=(int32_t)(r.ddr[a]|(r.ddr[a+1]<<8)|(r.ddr[a+2]<<16)|((uint32_t)r.ddr[a+3]<<24));
        if(got!=ref_gemm(As[t].data(),Bs[t].data(),ops[t].n,ops[t].k,i,j)) ++bad;
      }
  std::printf("%s\n", bad ? "*** WRONG ANSWERS ***" : "all four results match the host reference");
  return bad?1:0;
}
