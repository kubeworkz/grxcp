cmake_minimum_required(VERSION 3.20)
project(grxcp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# GRX-G100 sysroot discovery.
#
# GRXCP links against the INSTALLED sysroot only ($VORTEX_PATH), through
# pkg-config. Referencing the GRX-G100 source or build tree is a boundary
# violation (see AGENTS.md section 2) -- keep it that way so GRXCP builds
# against a released device stack, not against whatever is in someone's
# working copy.
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(VORTEX_RUNTIME REQUIRED IMPORTED_TARGET vortex-runtime)
pkg_check_modules(VORTEX_KERNEL           IMPORTED_TARGET vortex-kernel)

option(GRXCP_BUILD_TOOLS   "Build grx-smi / grx-prof / grx-sanitize" ON)
option(GRXCP_BUILD_LIBS    "Build grxBLAS and friends"               ON)
option(GRXCP_BUILD_TESTS   "Build the test suites"                   ON)
option(GRXCP_ENABLE_NPU    "Build the GRX930 NPU backend (phase 7)"  OFF)

add_library(grxrt SHARED)
target_include_directories(grxrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(grxrt PUBLIC PkgConfig::VORTEX_RUNTIME)

add_subdirectory(src)
if(GRXCP_BUILD_LIBS)  add_subdirectory(src/libs)   endif()
if(GRXCP_BUILD_TOOLS) add_subdirectory(tools)      endif()
if(GRXCP_BUILD_TESTS) add_subdirectory(tests)      endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS grxrt EXPORT grxcpTargets LIBRARY DESTINATION lib)
configure_file(cmake/grxrt.pc.in grxrt.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grxrt.pc DESTINATION lib/pkgconfig)
