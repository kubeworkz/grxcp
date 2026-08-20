// CUDA splits cg::reduce and the reduction operators into
// <cooperative_groups/reduce.h>. GRXCP has them in the one header, so this
// forwards. Including it is harmless and omitting it is not an error here --
// but a CUDA file includes it, and a file that has to delete the include to
// build is a file that was modified.
#ifndef GRX_COOPERATIVE_GROUPS_REDUCE_FORWARD_H
#define GRX_COOPERATIVE_GROUPS_REDUCE_FORWARD_H
#include <cooperative_groups.h>
#endif
