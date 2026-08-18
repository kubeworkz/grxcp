cmake_minimum_required(VERSION 3.20)
project(grxcp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GRXCP_BUILD_TOOLS      "Build grx-smi / grx-prof / grx-sanitize"       ON)
option(GRXCP_BUILD_LIBS       "Build grxBLAS and friends"                     ON)
option(GRXCP_BUILD_TESTS      "Build the test suites"                         ON)
option(GRXCP_ENABLE_NPU       "Build the GRX930 NPU backend (phase 7)"        OFF)
option(GRXCP_USE_MOCK_DRIVER  "Link the mock driver instead of libvortex"     OFF)

# ---------------------------------------------------------------------------
# GRX-G100 sysroot discovery.
#
# GRXCP links against the INSTALLED sysroot only ($VORTEX_PATH), through
# pkg-config. Referencing the GRX-G100 source or build tree is a boundary
# violation (see AGENTS.md section 2) -- keep it that way so GRXCP builds
# against a released device stack, not against whatever is in someone's
# working copy.
#
# GRXCP_USE_MOCK_DRIVER swaps the driver LIBRARY for the test fixture in
# tests/mock so the runtime can be built and exercised in CI without a sysroot
# or a simulator. It does NOT swap the HEADER: the mock implements the real
# vortex2.h ABI, so a genuine vortex2.h must still be on the include path.
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)

if(GRXCP_USE_MOCK_DRIVER)
  pkg_check_modules(VORTEX_RUNTIME IMPORTED_TARGET vortex-runtime)
  if(NOT VORTEX_RUNTIME_FOUND AND NOT GRXCP_VORTEX_INCLUDE_DIR)
    message(FATAL_ERROR
      "GRXCP_USE_MOCK_DRIVER is ON but vortex2.h was not found.\n"
      "Pass -DGRXCP_VORTEX_INCLUDE_DIR=<dir containing vortex2.h>, or make "
      "vortex-runtime.pc visible on PKG_CONFIG_PATH.")
  endif()
else()
  pkg_check_modules(VORTEX_RUNTIME REQUIRED IMPORTED_TARGET vortex-runtime)
endif()
pkg_check_modules(VORTEX_KERNEL IMPORTED_TARGET vortex-kernel)

add_library(grxrt SHARED)
target_include_directories(grxrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(GRXCP_USE_MOCK_DRIVER)
  target_sources(grxrt PRIVATE tests/mock/vortex_mock.cpp)
  target_compile_definitions(grxrt PUBLIC GRXCP_MOCK_DRIVER=1)
  if(GRXCP_VORTEX_INCLUDE_DIR)
    target_include_directories(grxrt PUBLIC ${GRXCP_VORTEX_INCLUDE_DIR})
  else()
    target_link_libraries(grxrt PUBLIC PkgConfig::VORTEX_RUNTIME)
  endif()
  message(STATUS "grxcp: MOCK driver -- results describe a synthetic device, not hardware")
else()
  target_link_libraries(grxrt PUBLIC PkgConfig::VORTEX_RUNTIME)
endif()

add_subdirectory(src)
if(GRXCP_BUILD_LIBS)  add_subdirectory(src/libs) endif()
if(GRXCP_BUILD_TOOLS) add_subdirectory(tools)    endif()
if(GRXCP_BUILD_TESTS) add_subdirectory(tests)    endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS grxrt EXPORT grxcpTargets LIBRARY DESTINATION lib)
configure_file(cmake/grxrt.pc.in grxrt.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grxrt.pc DESTINATION lib/pkgconfig)
