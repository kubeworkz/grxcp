# Runtime sources. The library target is declared at the top level so tools and
# tests can link it before this directory is processed.
file(GLOB GRXRT_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/backends/gpu_g100/*.cpp")

if(GRXCP_ENABLE_NPU)
  file(GLOB GRXRT_NPU_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/backends/npu_c930/*.cpp")
  list(APPEND GRXRT_SOURCES ${GRXRT_NPU_SOURCES})
endif()

if(NOT GRXRT_SOURCES)
  message(FATAL_ERROR "grxcp: no runtime sources found under src/runtime/")
endif()
target_sources(grxrt PRIVATE ${GRXRT_SOURCES})

# THE FLAG HAS TO REACH THE COMPILER, not just the glob.
#
# GRXCP_ENABLE_NPU switched which sources were compiled and nothing else, so
# npu_c930.cpp landed in libgrxrt while every `#ifdef GRXCP_ENABLE_NPU` block in
# src/runtime/context.cpp -- the probe, the device-table entry, the NPU property
# population -- compiled to nothing. A build with the flag on contained the
# backend and could not enumerate the device it drives.
#
# PUBLIC rather than PRIVATE: tests and tools that ask whether this build has an
# NPU need the same answer the runtime was compiled with.
if(GRXCP_ENABLE_NPU)
  target_include_directories(grxrt PUBLIC
                             "${CMAKE_CURRENT_SOURCE_DIR}/backends/npu_c930")
  target_compile_definitions(grxrt PUBLIC GRXCP_ENABLE_NPU=1)
  # And build what this directory defines. The glob above compiles npu_c930.cpp
  # into the runtime but never processed the CMakeLists next to it, so the
  # test_npu_c930 target declared there was never configured either.
  if(GRXCP_BUILD_TESTS)
    add_subdirectory(backends/npu_c930)
  endif()
endif()

# VX_types.h lives in the sysroot's KERNEL include directory, not the runtime
# one, and profile.cpp and tensormap.cpp both need it. Kept PRIVATE: a program
# using GRXCP should never need the driver's own headers.
if(VORTEX_KERNEL_FOUND)
  target_include_directories(grxrt PRIVATE ${VORTEX_KERNEL_INCLUDE_DIRS})
endif()
