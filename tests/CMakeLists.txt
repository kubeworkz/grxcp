enable_testing()

# Unit tests link libgrxrt. With GRXCP_USE_MOCK_DRIVER=ON that library already
# carries the mock driver, so these run anywhere; with it OFF they run against
# whatever backend $VORTEX_DRIVER selects.
file(GLOB UNIT_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
foreach(src ${UNIT_SRC})
  get_filename_component(name ${src} NAME_WE)

  # Some unit tests call the mock driver's inspection helpers (grxmock_*) to
  # assert on what the runtime SUBMITTED, not just on what it returned -- a
  # launch's grid dimensions, the argument blob's size. Those symbols exist only
  # in tests/mock, so with a real driver these do not link. Skipped by reading
  # the source rather than by a hand-kept list, because a hand-kept list is one
  # more thing to forget when a test starts using the mock.
  file(READ ${src} _src_text)
  if(_src_text MATCHES "grxmock_" AND NOT GRXCP_USE_MOCK_DRIVER)
    message(STATUS "grxcp: skipping ${name} -- needs -DGRXCP_USE_MOCK_DRIVER=ON")
    continue()
  endif()

  # A unit test may also ask a LIBRARY a question that needs no device kernel --
  # test_dispatch.cpp asks grxBLAS which engine a GEMM would land on, which is
  # pure host routing. Linked by what it includes, the same rule the library
  # gates below use, and for the same reason: a filename convention is one
  # rename away from silently not building a test.
  set(_unit_libs "")
  foreach(lib grxblas grxdnn grxfft grxrand grxsparse)
    if(TARGET ${lib} AND _src_text MATCHES "include[ \t]*<grx/${lib}\\.h>")
      list(APPEND _unit_libs ${lib})
    endif()
  endforeach()

  # A test that drives the GRX930 team's register model needs the model
  # compiled in. Read from what it includes, the same rule the library links
  # above use -- a filename convention is one rename away from silently not
  # building.
  set(_extra_src "")
  if(GRXCP_ENABLE_NPU AND _src_text MATCHES "npu_shim_adapter\\.h")
    set(_extra_src ${CMAKE_SOURCE_DIR}/third_party/grx930/npu_dpi_shim.c)
  endif()

  add_executable(${name} ${src} ${_extra_src})
  target_link_libraries(${name} PRIVATE ${_unit_libs} grxrt)
  target_include_directories(${name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/common
      ${CMAKE_SOURCE_DIR}/third_party/grx930)
  # A few unit tests include the driver's own VX_types.h to check GRXCP's view
  # of a descriptor field for field against the definition it has to match. That
  # header lives in the sysroot's KERNEL include directory, not the runtime one,
  # and it is deliberately not on grxrt's public interface: a program using
  # GRXCP should never need it.
  if(VORTEX_KERNEL_FOUND)
    target_include_directories(${name} PRIVATE ${VORTEX_KERNEL_INCLUDE_DIRS})
  endif()
  add_test(NAME ${name} COMMAND ${name})
  # Exit 77 is the skip convention: no device present is not a failure.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Library gates: tests/libs/test_<lib>.cpp is the numerical gate for library
# <lib>, and is built only if that library was. They need a compiled device
# kernel, which this build does not produce (see src/libs/CMakeLists.txt); the
# gate skips with 77 when the kernel is missing rather than reporting a
# numerical failure against a file nobody compiled.
#
# Which libraries a gate links is read from what it INCLUDES, not from its file
# name. The name is a bad key: test_grxblas_ex.cpp and test_grxblas_l12.cpp both
# test grxblas, and stripping "test_" gives two library names that do not exist,
# so both gates silently vanished from the cmake build. A gate that exercises two
# libraries at once -- test_libs_together.cpp does, deliberately -- has no single
# name to strip down to at all.
file(GLOB LIB_TEST_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/libs/*.cpp")
foreach(src ${LIB_TEST_SRC})
  get_filename_component(name ${src} NAME_WE)
  file(READ ${src} _src_text)

  set(_libs "")
  foreach(lib grxblas grxdnn grxfft grxrand grxsparse)
    if(TARGET ${lib} AND _src_text MATCHES "include[ \t]*<grx/${lib}\\.h>")
      list(APPEND _libs ${lib})
    endif()
  endforeach()

  if(_libs)
    set(_extra_src "")
    if(GRXCP_ENABLE_NPU AND _src_text MATCHES "npu_shim_adapter\\.h")
      set(_extra_src ${CMAKE_SOURCE_DIR}/third_party/grx930/npu_dpi_shim.c)
    endif()

    add_executable(${name} ${src} ${_extra_src})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/unit
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_SOURCE_DIR}/third_party/grx930)
    target_link_libraries(${name} PRIVATE ${_libs} grxrt)
    add_test(NAME ${name} COMMAND ${name})
    # RUN THEM WHERE THEIR REFERENCE VECTORS ARE.
    #
    # test_grxdnn_gelu, _attn and _block open gelu_ref.bin, attention_ref.bin
    # and block_ref.bin by bare filename -- they were written to be run from
    # tests/libs, which is how ci/run_real.sh runs them. Under ctest the working
    # directory is the build tree, the open fails, and all three report FAILED.
    # A gate that cannot find its own input has not measured anything, and
    # saying "failed" there is a false report of a wrong answer.
    #
    # Pointed at the source directory they find the reference, then find no
    # compiled .vxbin (this build does not produce one) and skip with 77, which
    # is the truthful outcome.
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libs)
  endif()
endforeach()

# The device-property contract must hold on an FPGA backend too, where managed
# memory has to be gated off. Same binary, different backend selection.
if(TARGET test_device_props)
  add_test(NAME test_device_props_fpga COMMAND test_device_props)
  set_tests_properties(test_device_props_fpga PROPERTIES
    ENVIRONMENT "VORTEX_DRIVER=xrt" SKIP_RETURN_CODE 77)
endif()

# ---------------------------------------------------------------------------
# The c930 RTL under Verilator (tests/rtl/)
# ---------------------------------------------------------------------------
#
# Built only when the GRX930 RTL is on hand and Verilator is installed:
#
#   cmake -DGRXCP_ENABLE_NPU=ON -DGRXCP_C930_RTL_DIR=/path/to/c930 ...
#
# Without either, the target simply does not exist. That is deliberate: a test
# that cannot be built should not be a test that reports 77, because a skip is a
# claim about the machine and "the RTL is not checked out" is a claim about the
# checkout.
#
# The RTL is NOT vendored. It is theirs, it moves, and a copy here would go
# stale silently -- which is exactly the failure third_party/grx930/README.md
# exists to prevent for the two files we do vendor.
#
# PARAMETERS ARE PASSED EXPLICITLY. c930_npu_top's own defaults are the "core
# defaults" of cuda_mapping.md 7.26 (8x8 array, MAX_M=64, MAX_K=256, MAX_N=8);
# c930_soc_top instantiates 4/4/8/16/12. Verilating with the defaults would
# build a different machine than the one NPU_C930_MAX_* describes.
if(GRXCP_ENABLE_NPU AND GRXCP_C930_RTL_DIR)
  find_program(VERILATOR_BIN verilator)
  if(VERILATOR_BIN AND EXISTS "${GRXCP_C930_RTL_DIR}/c930_npu_top.sv")
    find_package(verilator HINTS $ENV{VERILATOR_ROOT})
    if(verilator_FOUND)
      set(_c930_rtl
        c930_npu_top.sv c930_npu_core.sv c930_npu_csr.sv c930_npu_dma.sv
        c930_systolic_array.sv c930_tensor_pe.sv c930_fx_acc.sv
        c930_fp16_acc.sv c930_fp16_mul.sv c930_bf16_mul.sv
        c930_cla_sub.sv c930_cla_comp.sv)
      list(TRANSFORM _c930_rtl PREPEND "${GRXCP_C930_RTL_DIR}/")

      add_executable(test_npu_rtl ${CMAKE_CURRENT_SOURCE_DIR}/rtl/test_npu_rtl.cpp)
      target_include_directories(test_npu_rtl PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/unit
          ${CMAKE_CURRENT_SOURCE_DIR}/rtl)
      target_link_libraries(test_npu_rtl PRIVATE grxblas grxrt)
      verilate(test_npu_rtl SOURCES ${_c930_rtl}
               TOP_MODULE c930_npu_top
               PREFIX Vc930_npu_top
               VERILATOR_ARGS -Wno-fatal -Wno-lint -Wno-style
                              -GNUM_ROWS=4 -GNUM_COLS=4
                              -GMAX_M=8 -GMAX_K=16 -GMAX_N=12)
      add_test(NAME test_npu_rtl COMMAND test_npu_rtl)
      set_tests_properties(test_npu_rtl PROPERTIES SKIP_RETURN_CODE 77)
      message(STATUS "grxcp: building test_npu_rtl against ${GRXCP_C930_RTL_DIR}")
    else()
      message(STATUS "grxcp: verilator found but no CMake package; skipping test_npu_rtl")
    endif()
  else()
    message(STATUS "grxcp: no verilator or no c930_npu_top.sv; skipping test_npu_rtl")
  endif()
endif()
