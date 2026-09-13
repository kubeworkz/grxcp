// Core-only bench: drives c930_npu_core's data-plane port directly, so it can
// reach M=64/K=256 without the DMA's memory model. Checks C against a software
// reference and prints the three performance counters.
`timescale 1ns/1ps
module tb_core_m64 #(parameter int DW = 8);
  localparam int NR = 8, NC = 8, AW = 48;
  localparam int MAX_M = 64, MAX_K = 256, MAX_N = 8;

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic                   wen = 0, wsel = 0;
  logic [15:0]            waddr = 0;
  logic signed [DW-1:0]   wdata = 0;
  logic                   start = 0;
  logic [15:0]            dim_m, dim_n, dim_k;
  logic                   busy, done, error;
  logic [15:0]            c_raddr = 0;
  logic signed [31:0]     c_rdata;
  logic [31:0]            cyc, ops, stall;

  c930_npu_core #(
    .NUM_ROWS(NR), .NUM_COLS(NC), .DIN_W(DW), .ACC_W(AW),
    .MAX_M(MAX_M), .MAX_K(MAX_K), .MAX_N(MAX_N)
  ) dut (
    .i_clk(clk), .i_rst_n(rst_n),
    .i_wen(wen), .i_wsel(wsel), .i_waddr(waddr), .i_wdata(wdata),
    .i_staging_wen(1'b0), .i_staging_wsel(1'b0),
    .i_staging_waddr(16'd0), .i_staging_wdata('0),
    .i_bank_sel(1'b0), .i_wbank(1'b0),
    .i_start(start), .i_dim_m(dim_m), .i_dim_n(dim_n), .i_dim_k(dim_k),
    .i_precision(3'd0),
    .o_busy(busy), .o_done(done), .o_error(error),
    .i_c_raddr(c_raddr), .o_c_rdata(c_rdata),
    .o_cycle_count(cyc), .o_op_count(ops), .o_stall_count(stall)
  );

  integer a[0:MAX_M*MAX_K-1];
  integer b[0:MAX_K*MAX_N-1];
  longint cref[0:MAX_M*MAX_N-1];

  task automatic wr(input int sel, input int addr, input int val);
    begin
      @(negedge clk); wen = 1; wsel = sel[0]; waddr = addr[15:0]; wdata = val[DW-1:0];
      @(negedge clk); wen = 0;
    end
  endtask

  task automatic run_case(input int M, input int N, input int K);
    int errs, got, idx;
    begin
      dim_m = M[15:0]; dim_n = N[15:0]; dim_k = K[15:0];
      for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) begin
          a[m*K+k] = ($urandom % (1<<DW)) - (1<<(DW-1));
          wr(0, m*K + k, a[m*K+k]);
        end
      for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) begin
          b[k*N+n] = ($urandom % (1<<DW)) - (1<<(DW-1));
          wr(1, k*N + n, b[k*N+n]);
        end
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) begin
          cref[m*N+n] = 0;
          for (int k = 0; k < K; k++) cref[m*N+n] += a[m*K+k] * b[k*N+n];
        end

      @(negedge clk); start = 1;
      @(negedge clk); start = 0;
      wait (done);
      @(posedge clk);

      errs = 0;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) begin
          idx = m*N + n;
          @(negedge clk); c_raddr = idx[15:0];
          @(negedge clk);
          got = c_rdata;
          if (got !== $signed(cref[idx][31:0])) begin
            if (errs < 5)
              $display("  [FAIL] C[%0d][%0d] = %0d expected %0d", m, n, got, $signed(cref[idx][31:0]));
            errs++;
          end
        end
      if (errs != 0) begin
        $display("[FAIL] M=%0d N=%0d K=%0d : %0d mismatches", M, N, K, errs);
        $fatal(1, "mismatch");
      end
      $display("[M64] DW=%0d M=%0d N=%0d K=%0d  cycles=%0d ops=%0d stall=%0d  PASS",
               DW, M, N, K, cyc, ops, stall);
    end
  endtask

  initial begin
    repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);
    run_case(64, 8, 256);
    run_case(64, 8, 64);
    run_case(32, 8, 128);
    run_case(16, 8, 64);
    $display("[M64] all cases passed");
    $finish;
  end

  initial begin
    #200_000_000;
    $display("[M64] watchdog timeout");
    $fatal(1, "timeout");
  end
endmodule
