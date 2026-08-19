// Minimal reproducer: a tensor instruction issued from a second CTA.
//
// The kernel does one WMMA and nothing else. With a grid of one it completes;
// with a grid of two the simulator never returns.

#include <grx/device/grx_wmma.h>

namespace w = grx::wmma;
using elem = w::half;
using tile = w::tile<elem>;

struct tcu_repro_args {
  uint64_t a, b, d;   // fp16, fp16, fp32
};

__global__ void tcu_repro(tcu_repro_args* __UNIFORM__ arg) {
  w::fragment<w::matrix_a, tile::m, tile::n, tile::k, elem, w::row_major> fa;
  w::fragment<w::matrix_b, tile::m, tile::n, tile::k, elem, w::col_major> fb;
  w::fragment<w::accumulator, tile::m, tile::n, tile::k, float> acc;

  w::load_matrix_sync(fa, reinterpret_cast<const elem*>(arg->a), tile::k);
  w::load_matrix_sync(fb, reinterpret_cast<const elem*>(arg->b), tile::k);
  w::fill_fragment(acc, 0.0f);
  w::mma_sync(acc, fa, fb, acc);
  w::store_matrix_sync(reinterpret_cast<float*>(arg->d), acc, (unsigned)tile::n,
                       w::mem_row_major);
}
