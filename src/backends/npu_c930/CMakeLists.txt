# npu_c930 — GRX930 NPU backend for grxcp.
#
# This module drives the C930's INT8 systolic-array GEMM accelerator through
# its AXI4-Lite MMIO register interface.  It has NO dependency on vortex2.h.

# NOTE: the parent src/CMakeLists.txt globs *.cpp from this directory.
# test_npu_c930.cpp has main() and must NOT be compiled into libgrxrt.
# It is built separately below as a standalone test executable.
#
# The npu_c930.cpp is compiled into the grxrt library via the parent's glob.
# This file only defines the standalone test target.

add_executable(test_npu_c930
    test_npu_c930.cc
    npu_c930.cpp   # compiled separately for the test (not via the parent glob)
)

target_include_directories(test_npu_c930 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME npu_c930_unit COMMAND test_npu_c930)

# The decisions, driven through a register model: detection against a bus with
# nothing on it, and completion against a device that accepts a launch and never
# finishes. test_npu_c930.cc cannot reach either and says so.
add_executable(test_npu_c930_model
    test_npu_c930_model.cc
    npu_c930.cpp
)
target_include_directories(test_npu_c930_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME npu_c930_model COMMAND test_npu_c930_model)

# The fifth model, and the first one we did not write: the GRX930 team's own
# npu_dpi_shim, vendored under third_party/grx930. The four models above were
# built to produce a decision, which makes them evidence about the decision and
# none about the register map -- we invented both sides. This one carries their
# CSR addresses and computes a real INT8 GEMM, so the whole sequence is driven
# against someone else's idea of what these registers do.
#
# Still a model. See third_party/grx930/README.md for the five defects measured
# on import, and for why nothing green here says the c930 works.
add_executable(test_npu_c930_shim
    test_npu_c930_shim.cc
    npu_c930.cpp
    ${CMAKE_SOURCE_DIR}/third_party/grx930/npu_dpi_shim.c
)
target_include_directories(test_npu_c930_shim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/grx930
)
add_test(NAME npu_c930_shim COMMAND test_npu_c930_shim)
